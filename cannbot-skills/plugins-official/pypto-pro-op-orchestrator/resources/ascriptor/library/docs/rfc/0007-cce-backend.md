# RFC-0007: The cce backend and the launchers (M5)

Status: implemented (M5). Depends on: RFC-0001 (Lowered IR), RFC-0004 (board runner),
RFC-0006 (pass pipeline), D-013, D-015; the maintainer's five M5 requirements (2026-08-28) and the
local device isolation rule.

## 1. What the backend is

`ascriptor/backends/cce` prints a Lowered module as CCE source for the a5 family (`c310`,
`dav-3510`). It is the litmus test of IR completeness: one op prints as one statement, annotated
`// #<id>`, and anything the printer cannot express raises `CceGap` with the op, its id and its
source location. There is no fallback, no re-derivation of what a pass decided, no analysis:
addresses come from `addr_alloc`, flag ids from `events`, DMA parameters from `device_lower`,
sides from `split_sides`. Everything the printer does is table lookup and string formatting
(`arch/c310.py` holds the tables; `emit.py` the printer; `views.py` the byte-offset arithmetic of
the old handlers; `cpp.py` the spellings).

Three kinds of function print differently:

| IR function | printed as | code |
|---|---|---|
| `func @k.cube` / `@k.vec` (kernel sides) | `__aicore__ inline void k_cube(GM_ADDR x_, …, int32_t M, …, GM_ADDR workspace)` | one wrapper call per op (§2; every generated file opens with `using namespace ascrip;`, the position enum is `Position` (the DSL's `Position.L1`) because the compiler's vector intrinsics define a global `Pos`) |
| `vf @f` | `__aicore__ inline void f(__ubuf__ T* p, …, T s, …) { __VEC_SCOPE__ { … } }` | bare vector intrinsics on `vector_*` registers (§3) |
| `simt @f` | `__simt_vf__ __launch_bounds__(N) inline void f(__ubuf__ T* p, __gm__ T* g, T s)` | plain C on the compiler's SIMT layer (§4) |

Printed names are IR names made C identifiers (`emit.c_ident`, shared by the PTO ISA printer): other
characters become `_`, and a C++ keyword or reserved spelling (`__`, `_` and a capital), SIMT builtin, runtime
name, `<math.h>` name or name the kernel translation unit declares takes a trailing `_` (more while a function
repeats it); manifests keep `ir_name`. The op interface (manifest `name`, OpDef, tiling fields, aclnn binding,
harness) respells only keywords and runtime names (`cpp.api`): CANN's aclnn generator breaks an output name
ending in `_`.
The math names (`cpp.MATH_NAMES`: glibc 2.39's C++ `<math.h>` in every type suffix, plus libstdc++'s `abs`) are
reserved by decision: native Pro did not compile SIMT launches of `remainder` or `free` (A5-UP-042), which CCE
compiled. The PyPTO Pro printer gives vf and SIMT functions these names or `free` the same `_`
(I031).
`cpp.TU_NAMES` holds the functions and macros that a printed kernel's A5 compile declares, and `printf`: CCE did
not compile SIMT launches of `printf`, `abs`, `sqrt`, `max` or `vld`, since `launch<f>` needs a single `f`.
The entry cannot be respelled, since the custom-op build looks for the op type's spelling
(`runtime.opexec.aclnn_entry`): `compile_kernel` refuses such a kernel at its definition, before printing.

The artifact (`Artifacts.files`) is `<kernel>.cpp` (the `extern "C" __global__` entry:
`KERNEL_TASK_TYPE_DEFAULT`, `GET_TILING_DATA`, `pipe_barrier(PIPE_ALL)`, the scalar parameters read
from the tiling data, the per-core dispatch; its `GM_ADDR` tensor parameters come in the custom op's
tensor order — every input, then every output, each in signature order — because the op hands its
tensors over in that order, while the side functions keep signature order
(I028); its `workspace` argument is already the user region —
the CANN custom-op wrapper offsets past the system region before calling the entry, and the header
supplies the two `AscendC::` workspace functions that wrapper calls), `<kernel>_cube.h` / `<kernel>_vec.h`
(each opens with its `GMTensor` / `GMList` windows over the `GM_ADDR` parameters, and on c220 then
with `SetAtomicNone();` plus, on the vector side, `ResetMask(); SetMaskNorm();` — the launch SPR state
RFC-0008 specifies, established rather than inherited since nothing on that family clears these
between kernels), one header per `@vf` / `@simt` function,
`tensorutils_cce.h`, and `manifest.json` — parameters (kind, dtype, dims, output flag), workspaces
(sizes as host expressions over the scalar parameters), `block_dim` (`exported_block_dim` and
`exported_from` fix an import's launch, RFC-0015), mode, task type — which is all the runtime reads.

The manifest's `device` retains the selected profile and `arch` records that
profile's CCE architecture: C220 for A2/A3 and C310 for A5/A5PR. Both the returned
metadata and serialized manifest must agree with the architecture used to emit
instructions; a fixed default must not mislabel a successfully compiled artifact.

## 2. `tensorutils_cce.h`: the wrapper layer (requirement 1 and 2)

The header is new (`ascriptor/backends/cce/include/tensorutils_cce.h`, namespace `ascrip`); the
intrinsic sequences inside it are the old header's board-verified bodies; the API was reviewed by the
maintainer (D-041): wrapper names and parameter orders are the new ones, the window types keep the
old `[]` / `ptr()` shape, the buffer and event classes keep the old names as aliases, and the
helpers (cross-core, atomics, masks, scalar utilities) keep the old spellings.

The c310 scalar square-root instruction is spelled `::sqrt(float)` in kernel
and VF scalar code; SIMT uses `__sqrtf`. FP16 scalar square roots widen
through FP32 and explicitly narrow to the result type. A BF16 scalar square
root currently reports a located gap: the qualified c310 compiler rejects
the emitted plain scalar BF16 conversion. BF16 register casts are a separate
instruction domain and remain available (M10-055). Generic
`__builtin_sqrtf` is not an equivalent c310 spelling: it leaves an unresolved
`sqrtf` library symbol in the qualified CANN 9.1/9.2 custom-op builds
(M10-053). This is target instruction selection, without an algorithmic
fallback or host constant substitution. The qualified scalar input domain is
finite nonnegative floating values. Other domains require separate evidence.

A5 kernel/VF scalar `abs` uses `::abs` for i8/i16/i32/i64/f32, the intrinsic
selected by `AscendC::Std::abs` in the CANN SDK. A ternary comparison is not
equivalent for negative floating zero. FP16 widens/narrows through FP32;
BF16 conversion retains a located gap. Unsigned inputs are identity. SIMT
floating absolute value uses its own `__builtin_fabsf` spelling. The scalar
contract and integer representability boundary are in RFC-0001 §6.5.

* **Windows.** `GMTensor<T>` is a typed `__gm__` pointer; `Tensor<T, Pos>` is an absolute on-chip
  byte address with a compile-time position (`Position::UB | L1 | L0A | L0B | L0C | BT`). Both have the
  old window shape: `w[elems]` is the window `elems` elements further on, `w.as<U>()` the same
  bytes as `U`, `w.ptr()` the typed pointer of the window's position (`__gm__ / __ubuf__ / __cbuf__
  / __ca__ / __cb__ / __cc__ T*`; a number for the bias table), `load / store` the UB scalar
  accesses. `Buff<T, Pos, N>` is a slot buffer (`get(i)` = slot `i % N`, slot stride = the aligned
  slot size `addr_alloc` used) and `DBuff / TBuff / QBuff / PBuff` are its depth-2..5 aliases.
  Views are printed from their root: the printer folds the `mem.slice / get_buf / reinterpret /
  reshape` chain (`views.fold`) to a byte offset — ND row-major, NZ tiles as
  `(col / c0) * align16(rows) * c0 + row * c0 + col % c0` (c0 = 16 for L0C, else 32 bytes), packed
  4-bit dtypes in carrier bytes — and prints `root[elems]` (or `root.as<T>()[elems]`) when the
  offset is provably whole elements (`views.div_exact`), a byte-address window otherwise; one local
  per view.
* **Events.** `Event<SET_PIPE, WAIT_PIPE, PRESET, IDS...>`: static ids from the events pass,
  tokens rotate through the ids, `PRESET` tokens are set in the constructor and drained in the
  destructor; `SEvent / DEvent / TEvent / QEvent` are the depth-1..4 aliases the printer uses
  (`Event` beyond). `set / wait / set_all / release` print as the method calls. Autosync's events
  keep their IR names — `ev_<set pipe>_<wait pipe>_<ready | valid>_<n>` (D-043) — and the declaration's comment lists
  the buffers the event guards (`DEvent<PIPE_MTE2, PIPE_MTE1, 0, 0, 1> ev_mte2_mte1_ready_1;  // guards
  l1x, l1y #107`), so a reader pairs every set / wait with its buffer without the IR.
* **Folding (formatting only).** A scalar temporary the frontend named after its opcode (`add`, `mul.4`,
  `ceil_div.1`, `load`, `cube_idx` — a value the source never named) that is used exactly once, by a
  later op of the same block, is printed inside that use instead of as a `const` local; a value whose
  single use is the very next `scalar.set` folds whatever its name (`tile_cnt = tile_cnt + 1;`). Loop
  bounds are never folded (they would be re-evaluated per iteration). Between definition and use nothing
  may change what the expression reads: no control flow, no `scalar.set` / `scalar.cell`; an expression
  that reads memory (a load, or one folded into it) allows only pure declarations in between, a plain
  scalar expression is transparent to DMAs, syncs and stores. The folded ops' ids ride on the statement
  that absorbs them (`// #209 #210 #47`). User-named values keep their own statement, so the printed code
  follows the source (`emit.plan_folds`).
* **Cross-core mutex.** `sync.mutex` prints the old kernelbase protocol: the consumer side publishes
  `depth` tokens before the body (`CUBE_READY<PIPE_FIX>(id)` × depth for `vc`, `VEC_READY` for `cv`) and
  the producer side drains them before every `return` (`WAIT_CUBE / WAIT_VEC` × depth) — without the
  prologue the two sides wait for each other on hardware (found by T2, the interpreter models the tokens).
* **Helpers.** The old spellings: `CUBE_READY<PIPE>(id) / WAIT_VEC / VEC_READY / WAIT_CUBE /
  ALLCUBE_* / ALLVEC_* / INTRACORE_ALLVEC_*` (functions, the pipe a template argument),
  `SetFlag / WaitFlag / PipeBarrier<PIPE>`, `SetAtomicAdd<T> / SetAtomicMax<T> / SetAtomicMin<T> /
  SetAtomicType<T> / SetAtomicOpAdd|Max|Min / SetAtomicNone`, `SetVectorMask(hi, lo) /
  SetVectorMask(count) / SetVectorMaskByCount / ResetMask / SetMaskCount / SetMaskNorm`,
  `SetHF32Mode`, `DataCacheCleanAndInvalid` (a macro: the A5 compiler takes dcci's cache line and
  destination only as constant expressions, so they reach it as template arguments, I037),
  `CeilDiv / AlignUp / Min / Max` (f32 min/max print
  Pro's `max((float)(a), (float)(b))` outside VF and SIMT bodies, RFC-0001 §6.16), `GetCubeIdx /
  GetCubeNum / GetVecIdx / GetVecNum / GetSubBlockIdx`.
* **Ops.** One wrapper per opcode, named after it: `gm_to_ub_pad`, `ub_to_gm_pad`, `ub_to_ub`,
  `ub_to_l1`, `ub_to_l1_nd2nz`, `ub_to_l1_nz`, `gm_to_l1`, `gm_to_l1_pad`, `gm_to_l1_nd2nz`,
  `gm_to_l1_dn2nz`, `gm_to_l1_mx_scale_nd2nz`, `set_constant_to_l1`, `l1_to_l0<TRANS>`,
  `l1_to_l0_mx<TRANS>`, `l1_to_l0_img2col`, `l1_to_bt`, `mmad`, `mmad_bias`, `mmad_mx`,
  `mmad_mx_bias`, `l0c_to_gm_nz2nd / nz2nz / nz2dn`, `l0c_to_l1`, `l0c_to_ub`, the cross-core
  `cube_ready / wait_vec / vec_ready / wait_cube / allcube_* / allvec_* / intracore_allvec_*<PIPE>`,
  `barrier<PIPE>`, `set_flag_id / wait_flag_id`, the mask SPR helpers, atomics, `set_hf32`,
  `clean_dcache`, the core queries, and `simt::launch<fn>`. The tile coordinates in the IR (`dst_row0 /
  dst_col0 / src_row0 / src_col0`, mmad's `dst_row0 / dst_col0`) repeat the offsets of the operand's
  own view — `device_lower` copies `view.offsets` into them and the interpreter addresses through
  the view — so the printer's address is the folded view alone and the attrs are never added on
  top (found in the M5 review: the first printer added both); the wrappers take origins and extents
  only. An NZ fractal column is `align16(rows)` rows high in L1 / L0 (the L0 loads and the fixpipe
  address columns in 16-row units) and exactly `rows` high in UB: a `.nz()` view of a UB tile keeps
  the row stride the vector code packed it with (`v8_allhif8` packs P with a 33-row stride to dodge
  bank conflicts and `ub_to_l1_nz` receives it as `M_src`), which is also the interpreter's rule
  (D-044: aligning it to 48 read the second P slab from the wrong address on the board, not on the
  interpreter). Side-specific bodies sit under `if ASCEND_IS_AIC /
  ASCEND_IS_AIV` (constexpr) so the other unit compiles them to nothing.
* **Entry compatibility.** The `KERNEL_TASK_TYPE` / TLV / `g_coreType` / workspace block the CANN
  custom-op wrapper expects is kept verbatim with the CANN guards.

Requirement 2 holds by construction: kernel-level code never names an intrinsic; only `@vf` (and
the `@simt` helpers) do.

## 3. `@vf`: bare intrinsics

### Cast saturation and CTRL state (D-233)

`vf.cast.saturate` is the instruction's `RS_ENABLE` / `RS_DISABLE` operand, not a
global mode switch. On a5, `CTRL[60] = 0` selects that operand. With `CTRL[60] = 1`,
integer-destination conversions instead use `CTRL[59]`: **0 saturates, 1 truncates**.
Integer arithmetic's `CTRL[53]` is a separate control. `RoundMode` is independent of
saturation; integer-to-integer narrowing has no rounding operand.

The existing `core.set_sat_flag` / `core.get_sat_flag` vocabulary must expose `global`
for bit 60, alongside `float` (48), `float8` (50), `int` (53) and `cast` (59). These
APIs read and write raw bits: `enable=True` means a bit value of 1, which does **not**
mean saturation for every mode. The setter's `enable` attribute accepts a boolean
or an integer scalar value (zero clears, nonzero sets), so a saved flag can be restored.
Invalid mode names and non-scalar setter values are errors. The flags are kernel-level
operations; they must not be used inside a VF or SIMT body.

The caller must select the mode before calling the VF and complete outstanding vector
work before changing or restoring it (`barrier(Pipe.ALL)` is the conservative
recipe). Neither the printer nor a pass silently changes every kernel's entry mode.
`tests/kernels/a5/samples/cast_saturation.py` demonstrates saving both flags, the four
combinations of bits 60/59, per-instruction SAT/NO_SAT, and restoring the saved state.

The interpreter must track these flags per execution lane. A5 CTRL bits have no fixed
launch state (I013). Native Pro probes on AIV and AIC read bits 48/50/53/59/60 as
1/0/0/0/1 at entry. After a launch cleared bit 48, the next launch read 1 again. A bit 59
written by one launch survived into later launches and processes. The earlier CCE reading
of `global=1, cast=1` was therefore not a launch constant. Lanes still start from
`SAT_DEFAULTS` (`global=1, cast=1`, other bits 0), so runs stay deterministic, but that
start is a model choice.

The interpreter warns (`HardwareWarning`) when a lane uses a bit before writing it in the
same launch. A `core.get_sat_flag` read uses its bit. A read whose value only restores that
bit is exempt, because saving and restoring is the remedy. A `vf.cast` from a float, or
narrowing an integer, to an ordinary integer destination (below) uses bit 60, and bit 59
while bit 60 holds 1. The argument-free `i64 -> i32` form uses neither. The warning says that
the entry state is unknown on A5 and that the kernel should write the bit, and later restore
it, explicitly.

For ordinary integer destinations (`i8/u8/i16/u16/i32/u32/i64`), finite floating-point inputs are rounded first, then saturated
or truncated; integer inputs must remain in the integer domain, including unsigned
source values. Saturation clamps to the destination's range before narrowing. Masked
and layout-unselected lanes retain the existing zeroing/merging behavior. The argument-free
`i64 -> i32` form always discards high bits, including in global saturation mode
(board-measured at both int64 extrema). An explicit `saturate=True` on this form must
report a located gap; clamp explicitly before narrowing when saturation is required.
Non-finite float-to-integer inputs do not undergo modular truncation: NaN becomes
zero and infinities become the destination extrema even in NO_SAT mode (board-verified
for `f32 -> i16/i32/i64`). Float-destination and packed-codec saturation
remain separate semantics and are not claimed by this integer-conversion gate.

The gate is an independent numerical reference for the sample and boundary tests,
tracked interpreter goldens, pipe replay without hazards/deadlocks, backend emission
checks, and a bit-exact CCE board replay. The CTRL truth table was first measured with
both bare intrinsics and the installed AscendC `Cast` / `SetCtrlSpr` APIs, identical over
6400 bytes. The local cannsim gate sets `CPLUS_INCLUDE_PATH` to both the system C++
standard-library headers and their architecture-specific headers; without those, this
local compiler reports a missing `type_traits` (the setup issue recorded in D-232).
The official [Cast API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0398.html)
requires SetCtrlSpr to enable its SatMode; the [CTRL table](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_00091.html)
describes the selector and inverted bit-59 polarity. The installed a5 implementation
forwards SatMode to `vcvt` and leaves CTRL mode selection to the caller.

### Intrinsic printing

`vf.*` ops print as the compiler's vector intrinsics (`__clang_cce_vector_intrinsics.h`) with the
tag constants CANN's `dav_c310` register implementation passes: loads `vlds(dst, base, off, DIST)`
(`LOAD_DIST` maps `norm / ds_b8 / … / unpack4_b8`), stores `vsts(src, base, off, DIST, mask)`, block
copies `vsldb / vsstb`, unaligned `vldas / vldus / vstus / vstas`, masks `pset_bW(PAT_*)`,
`plt_bW(cnt&, POST_UPDATE)` on a counter cell or a copy of another value (RFC-0001 §5.2, I040),
`plds / psts`, `pand / por / pxor / pnot / pmov / psel / ppack /
punpack / pintlv / pdintlv`, compute `vadd / vmul / … (dst, a, b, mask, MODE_ZEROING)`,
`vcmp_* / vcmps_*`, `vsel`, `vdup / vbr`, `vci`, `vintlv / vdintlv`, `vselr / vgather2 /
vgatherb / vscatter`, `vsqz / vusqz`, `vexpdif / vmulscvt`, `dhistv2 / chistv2`, `vpack`,
`mem_bar(X)`, `sprclr(SPR_AR)`, and `vcvt` in the eleven argument shapes the compiler defines,
selected by the (dst, src) dtype pair (`CAST_SHAPES` — every pair of it compiles against the header in
`test_every_cast_pair_has_a_vcvt_form`; the unsigned <-> float pairs, narrowing to int8 and e8m0 <-> bf16
have no form on c310 and are gaps, D-054); a rounding mode the header's `static_assert` refuses for the
pair (`CAST_ROUNDS`: hif8 takes `ROUND_A` / `ROUND_H` alone, f32 -> e4m3 / e5m2 `ROUND_R` alone) and a
merging cast on one of c310's zeroing-only `vcvt` forms (`MERGE_REJECTED`: the integer widen / narrow
forms and float narrowing with a part — the compiler header admits `MODE_MERGING` for DAV_920R1 alone)
are named gaps, D-051 / D-054. Loads, stores and gathers cast to the
unsigned carrier register of the same width (the compiler declares those forms for the integer
element types). Absent masks print as `pset_b8(PAT_ALL)`. Loops inside `__VEC_SCOPE__` are
`for (uint16_t i = lo; i < hi; i += step)` — the compiler accepts no other induction type; a literal
step must be positive, a scalar step is printed as `(uint16_t)(step)` and taken as positive (D-059).

A scalar broadcast directly into a logical HiFloat8 register has no native C310
`vbr`/scalar-`vdup` overload. CCE reports this as a located gap (M10-038), including
zero. Code that intends a byte pattern can explicitly fill a UInt8 register and
reinterpret it as HiFloat8; numerical narrowing uses the declared float-register
cast with its explicit rounding mode. The printer must not invent quantization
or emit an invalid native broadcast.
The zeroing part of a masked HiFloat8 register copy is a bit operation: print
its zero register and selection through UInt8 carriers, preserving every active
source byte. This does not synthesize a numerical scalar conversion.
Register-to-register HiFloat8 broadcast likewise copies the lowest byte through
the UInt8 intrinsic; it does not reinterpret that byte as a scalar number.

A block copy (`vf.load` / `vf.store`, AscendC's `DataCopy<DATA_BLOCK_COPY>`) at stride 1 is the
contiguous `vlds` (zeroed to its mask) / masked `vsts`; a real stride is `vsldb` / `vsstb` with
`(stride << 16)` in the register's **own** element overload (`VSLDB_ELEM` / `VSSTB_ELEM`), never the
unsigned carrier: issued through the carrier — a reinterpret-cast of the register around the intrinsic —
the consumers read the register's previous value at the board's -O3 (D-052 / D-055, the rule bounded by
twenty board variants; D-057, the cause, found by running AscendC's own `LoadAlign<T, DATA_BLOCK_COPY>`
next to the printed form). A dtype without a native form (hif8) keeps the carrier and an idempotent
`vor` after the load; `tools/diag/probes/vsldb_hazard.py` re-checks all of it in one build.

64-bit registers are `vector_2xvl_s64 / u64` — `struct { vector_s32 / u32 val[2]; }`, filled by a
`vlds` with `DIST_DINTLV_B32`, so `val[0]` holds the low halves and `val[1]` the high halves (D-049).
The compiler header's `__VF_*_B64` macros give them add / sub (carry builtins), mul, div, mod,
max / min, and / or / xor / not, neg, abs, shifts (a `vector_s32` amount), the `*s` scalar forms,
cadd / cmax / cmin (no mode argument), sel, dup / br / mov, cmp, cvt and gather2 / scatter with a
`vector_u32` element index; the printer whitelists exactly those. Signed `abssub`, `muladddst`
and `axpy` use two-instruction sequences with a per-op temporary. Unsigned `abssub` is the
approved compiler extension `max(a, b) - min(a, b)` (three instructions), preserving all64
unsigned bits; its model must order operands before subtracting signed bit carriers (M10-035).
The printer takes `.val[0]` of a 64-bit index
register, and gathers 64-bit `gatherb` data as two 32-bit gathers by byte offset. A 64-bit register
is 32 lanes (the DSL's, the interpreter's and CANN's own `vector_int64_t = vector_s64`): the load is
the single-register `vlds(…, NORM)` deinterleaved with `vdintlv` into the two halves, the default
mask on a 64-bit operand is `pset_b32(PAT_VL32)`, and a store is one 32-bit `vsts` of the halves interleaved back (the header's two-register
`vsts` issues a second 256-byte store behind them, and the tile after the stored one read as zeros) — a bare 2xvl value would move 512 bytes and drag 32 garbage lanes into reductions and
scatters (D-050, found on the board). `vcmax` / `vcmin` leave the extremum's index in lane 1 on every
width, so the printer masks the result to lane 0 (the DSL's value-then-zeros register) unless
`index = true` asks for the bare instruction (a gap on 64-bit registers); a
64-bit scatter is AscendC's `ScatterImplB64` shape — the halves interleaved back into memory
order, (2 i, 2 i + 1) index pairs, one 32-bit `vscatter` — rather than the header's two-register `vscatter` macro; 64-bit `gatherb` is the
native single-register block gather. Complex registers
travel in their carriers (`c32` → `vector_u32`, `c64` → `vector_2xvl_u64`); add / sub / mul / div are
printed on the (re, im) parts — the fp32 halves of the c64 load, or, for c32, fp16 parts `vdintlv`-ed out of the packed lanes for add / sub and fp32 parts widened
straight from the lanes (`PART_EVEN` / `PART_ODD`) and rounded once back for mul / div — the
single rounding of torch's ComplexHalf, which the goldens carry (unmasked). Other ops on 64-bit or complex registers
(exp, sqrt, prelu, arange …) remain gaps.

### 3.1 Register groups (D-229)

The preceding D-050 description is the `n = 1` ABI. With `n = 2`, all 64 lanes of
the integer/c64 two-register carrier are live and normal loads/stores transfer
512 bytes. Masks, gathers/scatters, casts, reductions and the interpreter must
agree on that width. The c32 group contains 128 complex-half elements. CCE's
compiler-provided overloads supply the integer arithmetic sequences; a printed
overload is not a claim of one hardware instruction. The printer must never
implement a two-register request by keeping only its first register.

The support gate includes independent numerical checks and interpreter-recorded
sample goldens, the pipe simulator, CCE compilation, cannsim and the board. Cases
must make the upper half observable and include a partial mask crossing the
single-register boundary, an aliased destination, integer carries, conversion,
reduction and interleave. The one-register corpus remains a regression gate.

Gate completed 2026-09-05: six `samples/reg_groups.py` kernels are bit-exact on
the interpreter, pipe simulator, cannsim and board; their twelve vec/cube units
compile. The full suite passes (2640 passed, 23 skipped), and the five canonical
one-register kernels and cast matrix pass the explicit local compiler tests.

Integer decreasing arange lowers to increasing indices, negation and the starting
value: CCE's `DEC_ORDER` reverses an increasing interval, whereas the DSL computes
`start - lane`. Its first lane must equal the requested start, including when the
start cannot be represented exactly in float64.

## 4. `@simt`

`simt.*` ops are C: `thread_id / thread_num / blk_idx / blk_num` from the compiler's builtin
variables, loads and stores as pointer indexing, `simt.atomic{op}` as `ascrip::simt::atomic_*` — add /
sub / max / min / exch / and / or / xor over the `atomicAdd … atomicXOr` builtins, u32 inc / dec over
`atomicInc` / `atomicDec`, and cas over `atomicCAS` with its compare operand (D-059; the builtin returns
the prior value for i32 / u32 / f32 and GM i64 / u64, and other types are fire-and-forget plus a readback
without that guarantee, as CANN's dav_3510 does),
`simt.barrier` as `__sync_workitems`. The launch prints as `ascrip::simt::launch<fn>(threads, …)` =
`cce::async_invoke<fn>(dim3{threads, 1, 1}, …)` on the vector unit. A function's
`__launch_bounds__` is the largest thread count of its launch sites.

Native A5 integral-rounding builtins can canonicalize a negative zero. The
existing SIMT compatibility wrapper adapts only zero-magnitude output bits to
the sign required by RFC-0001, using integer fields. This target-specific
spelling repair belongs alongside the native builtin, rather than introducing
A5 compiler behavior into the shared mathematical IR (M10-054). It preserves
nonzero/NaN result bits; no software rounding algorithm is substituted.
`simt::fmod` has no rounding step: like PyPTO Pro's printer, it forms the exact
remainder by integer long division of the operand fields and returns quiet NaN
0x7FC00000 for a NaN operand, infinite dividend or zero divisor (I022).

## 5. Gaps are errors

`CceGap` carries the op, its id and its `loc`. The corpus-wide list is in
`tests/kernels/a5/corpus.json` (`cce_xfail`, with reasons); `tests/backends/test_cce.py` fails
when a listed kernel starts printing or an unlisted one stops. Ops the corpus never uses but the
registry has (`dma.gm_to_ub.nd`, `list.*`, `vf.log2 / log10`, `atomic` fixpipe stores) are gaps
with the same mechanism. Two semantic notes: `scalar.div` prints as C `/` (the interpreter uses
Python floor division; the corpus divides non-negative sizes), and `vec.set_mask_count` has no
c310 builtin (a call fails at compile time by `static_assert`).

## 6. Gates

* **T1** — `tests/backends/test_cce.py`: every corpus kernel prints or is listed; every op id
  appears once in the sources; the manifest is readable by the runtime; the five canonical kernels
  compile in both units with the local bisheng when it is present (`ASCEND_HOME_PATH`,
  `ASCRIPTOR_GXX_INCLUDE` for a clang that cannot find a C++ standard library). **A gate that
  skips reports nothing**, and this one skipped for months on any box with a distribution
  toolchain: the test spelled the architecture-specific include root itself as
  `$ASCRIPTOR_GXX_INCLUDE/x86_64-conda-linux-gnu`, which a distribution keeps in a sibling tree
  (`/usr/include/x86_64-linux-gnu/c++/11`) instead, so the `<type_traits>` probe failed on
  `bits/c++config.h`, `_bisheng_works` read that as "no compiler", and every check behind it
  skipped silently. `ascriptor/runtime/build.gxx_arch_roots` already knew both layouts and is
  now the single owner. Turning the gate back on immediately failed the cast-pair check — for a
  second missing include root, not a cast gap: `tensorutils_cce.h` includes `scalar_math.h`,
  which sits beside it only in a compiled artifact, and the resulting `fatal error` names that
  header rather than the generated file, so it survived the per-pair filter and showed up only
  in the return code. Both are fixed; when adding a bisheng-gated check, confirm it runs before
  trusting that it passes. The corpus-wide
  bisheng check (`tmp/m5/check_corpus.sh` during M5): every printed kernel compiles in both units
  (102 of 103 at M5, all since the 64-bit / complex forms of D-049).
* **T2** — `ascriptor run kernel.py::name --case <golden> --launcher cannsim`: the aclnn package
  built with the local CANN and run under `cannsim record`, outputs compared with the functional
  golden — bit for bit first; when that differs, numerically within the corpus `replay_tolerance`
  entry or the `--rtol/--atol` given, and the accepted bitwise differences are reported as notes so
  "bit-exact" and "within tolerance" stay distinguishable (the cube's fp32 accumulation order is
  not the interpreter's). Result (2026-08-28, local CANN 9.0 cannsim, `tools/run_cases.py --launcher
  cannsim --rtol 1e-4 --atol 1e-5` over the canonical kernels that have goldens): `matmul_float_bias_bt`,
  `matmul_float_mmad`, `simt_axpy` within tolerance (at most 2.9e-6 absolute / 1.1e-4 relative),
  `bf16_to_fp4_e1m2` bit-exact; after the API review (D-041, folded printer): `bias_splitn` (split-N
  with a bias row, sliced NZ tiles), `matmul_mknk_2dgrid_splitn` (the entry-name rule) and
  `matmul_abs_add1_vf` (cube + @vf, DBuff, cross-core mutex) within tolerance, `bf16_to_fp4_e1m2` still bit-exact; `vec_cube_abs_sqrt_matmul` (@vf sqrt/abs/cast
  to half, then the cube) runs to completion once the mutex prologue is printed and matches to 2.2e-3
  absolute — the 1 % of elements outside the fp32 tolerance sit in 192 of 4096 rows, the signature of
  a half-ulp difference in the vector unit's `vsqrt` / `vcvt` on a few inputs carried along the row,
  not of an addressing or sync fault (the two extra goldens are untracked: 3 MB and 18 MB). What the
  gate found that T1 cannot: the custom-op wrapper hands
  the entry the *user* workspace region (§1); bf16 / fp8 arguments must reach the harness with their
  logical shape, not the byte carrier's; the first printer displaced sliced NZ tiles twice (the view
  and the `row0 / col0` attrs, §2); the matmul bias row must be sliced as ND; and the custom-op build
  names the kernel file and symbol by its own op-type snake rule (`runtime.opexec.aclnn_entry`).
  Practical note: one `cannsim record` takes ~4.5 GB, so T2 sweeps run one kernel at a time on a
  16 GB box.
* **T3** — the same with `--launcher board` on the shared box of `machine_specs.md` (§7). Result
  (2026-08-28, the first day the box was reachable): the T2 set within the fp32 tolerance —
  `matmul_float_mmad` 1.4e-6, `matmul_float_bias_bt` 2.9e-6, `bias_splitn` 3.8e-6, `simt_axpy`
  9.5e-7, `matmul_mknk_2dgrid_splitn` 3.4e-5, `hif8_carrier_matmul` 7.6e-6, `matmul_abs_add1_vf`
  2.7e-5, `v8_allhif8` 7.6e-6 (after D-044) — and `bf16_to_fp4_e1m2` bit-exact; a cached build runs
  in about a minute. What T3 found that T2 could not: the printer's NZ column height for UB `.nz()`
  views (D-044; the interpreter and the old simulator agree with the board, cannsim agreed with the
  board only on the whole kernel) and the runner's own three defects (§7).
* **64-bit and complex registers** (D-049 / D-050, 2026-08-28): the 13 former `cce_xfail` kernels
  and the two samples that exercise the same forms (`reduce_family`: `cadd` / `cmax` / `cmin` on
  fp32, int32 and int64 rows; `gatherb64`: the native block gather). cannsim: `int64_fused`,
  `int64_safe`, `int64_shift`, `int64_elt`, `int64_reduce`, `u64_core`, `shiftl_i64`, `shiftr_i64`,
  `gather_b64_u64`, `scatter_b64_u64`, `scatter_b64_u32`, `c32_arith` and `gatherb64` bit-exact,
  `c64_arith` (the division, 2.4e-7) and `reduce_family` (one fp32 sum byte, 3.8e-6) within
  tolerance. Board: the same verdicts — every one of those bit-exact, `c64_arith` and `reduce_family`
  within the same tolerance. What the board found that cannsim hid (D-050): the 2xvl forms are 64
  lanes — bare 2xvl loads and stores moved 512 bytes and dragged garbage lanes into reductions and
  scatters; `vcmax` / `vcmin` leave the extremum's index in lane 1 on every width; the header's
  two-register `vsts` issues a second 256-byte store behind the first and zeroed the scatter kernels'
  index tile (a cannsim register probe showed the index register empty while the data register
  was right). What the goldens settled for c32: torch's ComplexHalf rounds once from fp32, so the
  fp16 four-product sequence was 7.8e-3 off and the printer computes c32 mul / div in fp32.
* **Support-surface samples** (D-045…D-047, 2026-08-28): one sample kernel per printed feature,
  its golden recorded from the interpreter after a torch-reference unit test under `tests/sim/`
  validates the interpreter's semantics, then T1, the pipe-level simulator, cannsim and the board.
  `vf_log_family` within the fp32 tolerance (the `vln` + `vmuls` product); `nd_dma_pad` (constant /
  nearest padding, a three-loop gather, the transpose sugar, bf16 config pads) bit-exact on cannsim
  and on the board — the AscendC NdDma contract read from CANN's `DataCopyWithNDDMAImpl` held;
  `sort_family` (`sort32`, `mergesort4`, `mergesort_2seq` over fp32 records) bit-exact on the board
  and on cannsim — `vbs` / `vmrgsort4` produce exactly the interleaved (score, index) records the
  old simulator modelled, index halves included; `list_concat` (a `GMList` parameter through the
  aclnn dynamic input, D-048) bit-exact on the board and on cannsim after one round trip: the first
  run copied only the first member because the wrapper took the descriptor's per-member header
  word for the count, as the local CANN headers' `ListTensorDecode` would — the probe (§7) showed
  that word is 1 for every member on both launchers, so the count is now derived from the
  pointer-array offset.

## 7. Launchers and `OpExec` (requirements 3, 4, 5)

`ascriptor/runtime`:

* `project.py` — the aclnn custom-op project from an artifact (D-013: the template directory
  `runtime/aclnn/template/` replaces the old tars); `op_host/<kernel>.cpp` (tiling from the scalar
  attributes, block dim from the manifest or the platform's core count, dynamic UB for SIMT on
  non-cube kernels, workspace = user bytes from the manifest expressions + the platform's system
  workspace, infer-shape / infer-dtype for the outputs, the OpDef), `op_host/<kernel>_tiling.h`,
  `CMakePresets.json` with the CANN path and compute unit.
* `harness.py` — the host program; it reads `input/args.txt` (shapes and scalars) and the input
  `.bin` files, so a new shape never rebuilds anything; outputs are poisoned (0xFF) before the
  launch, as the interpreter poisons them (D-026).
* `gmlist` parameters (D-048; outputs D-060): the op declares the parameter `ParamType(DYNAMIC)` —
  an `Input`, or an `Output` when the kernel returns the list — the aclnn API takes an
  `aclTensorList` for it in either position, and the framework packs the members into AscendC's
  ListTensorDesc in GM before the launch — the kernel receives one `GM_ADDR` and reads the
  descriptor with the `GMList<T>` wrapper (RFC-0001 §13). `args.txt` carries `L name count` and
  one `T name.j ...` line per member; an input list's bytes come from `input/name.j.bin`, an
  output list's members are created from their shapes, poisoned like a tensor output and saved
  as `output/name.j.bin` each (`OpExec` returns the list of members, as `run_kernel` does; the
  goldens record them one by one). A DYNAMIC output shifts the instance index of every output
  after it, so the generated infer-shape / infer-dtype address the outputs through
  `GetIrOutputInstanceInfo` when a list output exists; infer-shape leaves the members' shapes
  alone — they are the caller's tensors', ragged dims included, and that is what the descriptor
  carried on cannsim and on the board (`list_split`, bit-exact on both).
  The descriptor's per-member header word is not what the local CANN headers' decoder expects
  (its high half is 1 for every member on the board and under cannsim), so the wrapper derives the
  count from the pointer-array offset instead — the two probes are `tmp/list/probe_desc*.py`
  (git-ignored): a printed kernel patched to copy the first 512 descriptor bytes into its output.
* `build.py` — `build.sh` + the `.run` install, the harness compile, the run (`./test_aclnnop` or
  `cannsim record ./test_aclnnop -s <chipset>`), all under an advisory `flock`. **No idle check**
  (requirement 3): the lock serialises this workspace's runs; other users' processes are the
  operator's responsibility to inspect.
* `opexec.py` — `OpExec(kernel, launcher="aclnn" | "cannsim" | "sim" | "board", out_dir=…,
  cann_path=…, custom_op_path=…, block_dim=…)`; `aclnn` is the default (requirement 5). Sources
  are regenerated per call, the package rebuilt only when their hash changes.
* `board.py` — RFC-0004's local device environment, lock and run path for one
  `OpExec` call. `boards.json` is ignored machine configuration; its selected entry
  must say `"local": true` and provide a local workspace and device identity.
  Connection fields are rejected. The runner builds and executes on this machine
  under the workspace lock, with isolated output for each task.

`ascriptor compile kernel.py::name -o dir` writes the artifact; `ascriptor run kernel.py::name
--case <golden> --launcher …` runs a golden case and prints the bitwise verdict.

## 8. Not in M5

The full record of what the printer covers and what it reports as a gap is `docs/cce-support.md`
(generated by `tools/cce_support.py`, checked by the T1 tests). In short:

`vec.topk_radix` (AscendC's `TopK` library, for the ascendc backend), the direct (non-aclnn)
launcher, batch bundles of many cases per transfer (RFC-0004 §2 step 1 — `OpExec` ships one kernel
per call), the c220 tables (M9). Done after M5 (2026-08-28): `vf.log2` / `log10`, the ND DMA, the
sort family, `gmlist` parameters, the 64-bit and complex register forms (D-046 … D-049).
