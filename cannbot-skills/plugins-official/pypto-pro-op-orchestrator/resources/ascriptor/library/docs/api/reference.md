# Public API reference

The [manifest](manifest.json) is the explicit compatibility inventory for `0.1.x`. Its
declaration paths resolve to `.pyi` files shipped in the package. Each name has a semantic
description and a coverage disposition; aliases may share the same example. Emission,
functional simulation, pipe simulation, vendor compilation and board execution are separate
verification stages.

## Device modules and decorators

| Module | Profile | Vector body | Decorators |
|---|---|---|---|
| `ascriptor.a2` | `b3` | Kernel-level UB tensor instructions | `kernel`, `func` |
| `ascriptor.a3` | `a3` | Same functions/signatures as A2 | `kernel`, `func` |
| `ascriptor.a5` | `950` | Register instructions inside VF; scalar SIMT | `kernel`, `vf`, `simt`, `func` |
| `ascriptor.a5pr` | `950pr` | Same source vocabulary as A5 | `kernel`, `vf`, `simt`, `func` |

`@kernel` and `@kernel(mode="vec", block_dim=1)` bind a kernel to its module's profile.
`mode` is `mix`, `vec` or `cube`. `block_dim` is an integer or an expression over signature
symbols; see [RFC-0002](../rfc/0002-frontend-static-subset.md). The decorated entry is not
a host callable: inspect `.ir()` or use `OpExec`. `@vf` has no extra parameters;
`@simt(num_threads=...)` accepts 64, 128, 256, 512, 1024 or 2048 threads.
`@func` is a source-inline helper. The empty-call decorator forms are retained.

Shared facade imports retain established spellings that a device may not implement.
Importability is not device support: A2/A3 do not provide VF or SIMT decorators, and A5
register forms cannot be used as A2 tensor-vector forms. The intentional A2 replacements
are enumerated in `frontend/dsl_vec.pyi`.

`KernelFn`, `VfFn`, `SimtFn`, `InlineFn`, `DTypeName`, `EnumValue`, `GMSpec`,
`make_decorators`, `marker` and `rule_of` are implementation helpers, excluded from facade
exports. They remain readable/importable from their implementation modules. The stub's
private result types describe expressions without adding facade exports.

## Signatures, scalars and ordinary control flow

`GM[dtype, (dims, ...)]` declares a tensor parameter. Each dimension is an integer, a string
symbol or a multiplication-only symbol expression. Repeated symbols must agree across all
inputs, outputs and explicit scalar bindings. `GMList[dtype, ("?", "D")]` permits a member
dimension that varies; `GMList[dtype, ("?", "D"), 8]` pins the count. Members expose `.shape`;
the list exposes `.count`, `len`, indexing and iteration. Bare `GMTensor` and `GMTensorList`
remain diagnostic names and fail with a replacement suggestion.

Kernel parameters are inputs, outputs, then explicit scalars. Returned parameter names own
the output ABI. Scalar annotations use dtype names (`n: i32`, `alpha: f32`); `Var` as an
annotation means `i32`. An in-place or accumulating output needs an explicitly initialized
buffer; a full-overwrite example poisons output elements before execution.

`Var(value=None, dtype=None, *, name="")` is a mutable scalar. Supply a value or dtype —
by position the first argument is the value, so a dtype alone is `Var(dtype=i32)`, not `Var(i32)`.
Mutation uses `.set(value)` or augmented assignment, while `=` binds a new name and cannot
silently replace a mutable cell. `.GetValueFrom(view)` and `.SetValueTo(view)` load/store a
scalar through a one-element tensor view at kernel level: the kernel body or an
inlined `func`, never a `vf` or `simt` body, and a wider view silently resolves
to its first element (see [authoring](authoring.md#scalar-values-and-memory)). Arithmetic, comparisons and
bitwise operators build scalar IR. `CeilDiv`, `Min`, `Max`, `Align8` through `Align256` and
the `var_*` spellings preserve the same scalar instruction vocabulary.

`range` is a device loop, including with constant bounds. `unroll` takes one to three static
integer bounds and expands the body. Ordinary `if`, `break`, `continue` and supported
augmented assignments preserve the frontend's region/dataflow checks. Dynamically assigned
branch locals do not escape their region. The [scalar example](../../examples/api/scalar_control)
observes both zero-trip and early-exit behavior against a Python arithmetic reference.

## Dtypes, descriptors and enums

`DT` includes the established dtype names and neutral aliases (`f16`, `bf16`, `f32`, `i32`,
and the complete list in the manifest). A dtype has `.bits`, `.size` in bytes, and `.C0`, the
elements per 32-byte block. Packed four-bit formats have no `.size`; `.C0` is 64. `DT.f32`
and `DT.float` are the same descriptor. Top-level convenience aliases retain their identity.

`Position` names GM, L1, L0A, L0B, L0C, UB and BT. `Pipe` names hardware queues and ALL;
`VfPipe` names local VF memory-barrier participants. `Layout` carries ND/NZ hints.
`RoundMode`, `RegLayout`, `MaskMergeMode`, `MaskType`, `CompareMode`, `DualMode`, `PostMode`,
`HighLowPart`, `LoadDist`, `StoreDist`, `HistBin` and `HistMode` contain only the manifest's
declared members. `SelectMode` belongs to the A2/A3 tensor-vector facade. They are static
descriptor values, not integer enums; pass the named member required by an instruction.

`CastConfig(round_mode=RoundMode.TO_EVEN, reg_layout=RegLayout.ZERO, saturate=False,
name="", merge_mode=MaskMergeMode.ZEROING)` describes an A5 register cast. It does not
change the global CTRL flags. `Conv2D(kh, kw, pad=(left,right,top,bottom), stride=(h,w),
dilation=(h,w))` is static geometry; `.out_hw(h,w)` computes the output spatial dimensions.

## Tensors, views and buffers

`Tensor(dtype, shape, position=Position.L1, name="", layout=None)` allocates a two-dimensional
on-chip tensor. GM storage belongs in parameters or workspace declarations. Default cube
memory layouts are NZ; UB defaults to ND. `.shape`, `.span`, `.offset`, `.dtype`, `.position`,
`.name` and `.is_transpose` describe the compile-time view.

Indexing/slicing and `.reshape`, `.flatten`, `.reinterpret` create views. Offsets and strides
in `.view(shape, strides, offset=0)` are element units. Byte widths still determine physical
DMA footprints and packed carriers. Negative view extents/strides are rejected. A non-unit
innermost stride uses the A5 ND DMA read path; the corresponding GM store is a located gap.
Composed view restrictions are specified in [RFC-0010](../rfc/0010-gm-strided-views.md).
`.T` is a source rider on supported cube/GM copy paths; it does not transpose allocated
bytes. UB `.nz()`/`.nd()` describe layout views. `.set_shape()` is unsupported.

`<<=` copies memory or registers according to their spaces. UB load riders `.single()`,
`.brcb()`, `.upsample()`, `.downsample()`, `.unpack()` and `.unpack4()` choose register load
distributions. L0C riders `.relu()`, `.requant(scale, offset, hif8_hybrid)` and `.subblk(id)`
select supported fixpipe behavior. A requantized L0C-to-UB copy needs an explicit subblock;
dtype-changing split copies are rejected. [The strided-view example](../../examples/api/strided_views)
compares every logical output element using independent indexing.

`DBuff`, `TBuff` and `QBuff` take Tensor's allocation arguments plus the keyword-only
`sync_depth=None`, and contain two, three or four physical slots. `buffer[index]` selects modulo
the physical slot count. On A2/A3, `sync_depth=N` keeps those addresses unchanged while making N the
credit count of the slot session that guards the allocation, in both of its directions; it must be
in `1..slots`. It does not alter explicit events, cross-side mutex credits or storage capacity.
`GMBuff(dtype, shape, slots, name="",
per_core=True)` is a workspace ring; its lowering checks one beat counter, bounded reader lag
and cross-side mutex coverage. `split_workspace` is an ordinary GM workspace view. Allocation
does not prove lifetime safety. Run [the ring example](../../examples/api/buffer_ring)
as `python buffer_ring.py check --launcher pipesim`, or call
`check(launcher="pipesim")`, to check wraparound, events and GM hazards. Its default
`check()` uses functional simulation and reports that stage separately.

## Synchronization, DMA and cube instructions

`with auto_sync():` derives the same-side synchronisation for the supported on-chip memory
dependencies: physical-slot mutexes on A5, paired `ready`/`valid` slot sessions on A2/A3, with no
strategy option between them (RFC-0005 §5). `with cube_scope():` and `with vec_scope():` assign
operations to a side of a mixed kernel. `SEvent` through `QEvent` are one- through four-slot
events with `.set()`, `.wait()`, `.setall()` and `.release()` methods.
`setflag(src, dst, event_id)`, `waitflag(src, dst, event_id)`, `barrier(pipe)` and `bar_*()`
are explicit pipe synchronization. They are kernel-level operations.

`VcMutex(id, *, depth=None, guards=None, ...)` and `CvMutex` order vector-to-cube and
cube-to-vector handoffs; one of `depth` or `guards` is required and both are keyword-only, and
the optional `src_start_pipe`, `dst_start_pipe`, `src_end_pipe`, `dst_end_pipe` are keyword
arguments. Use `.lock()`/`.ready()` on the producer and `.wait()`/`.free()` on the consumer.
All four are required, and the consuming instructions belong BETWEEN `wait` and `free`: a `wait`
followed straight by `free` returns the slot before anything read it, and a read outside that
pair is ordered by nothing however many mutex calls the loop contains
([cross-side ownership](synchronization.md#cross-side-ownership) has the sequence).
`depth` is a credit count, not a pipelining hint: the consumer publishes that many credits up
front, so `.lock()` of cycle `i` blocks on the `.free()` of cycle `i - depth` and not at all
while `i < depth`. It must not exceed the number of SLOTS the handed-over buffer has — one
mutex cycled twice over one tile needs `depth=1`, and `depth=2` belongs to a buffer that hands
over a different slot each cycle. `crosssync` refuses the mismatch where it can prove nothing
else orders the two cycles ([M10-076](synchronization.md#cross-side-ownership)).
The `cube_ready`/`wait_cube`, `vec_ready`/`wait_vec`, `allcube_*`, `allvec_*` and
`intracore_allvec_*` families expose cross-core flags; flag IDs must be static integers in
0..7 and pipe constraints depend on the device. Cross-side teaching coverage remains in the
next example batch; this declaration does not establish a safe protocol by itself.

Explicit `gm_to_*`, `ub_to_*`, `l1_to_*`, `l0c_to_*`, `set_constant_to_l1`
and `gm_to_ub_nd_dma*` parameters are individually listed in the stubs. Names ending in
`_element` use element units; ordinary block/repeat strides use 32-byte blocks where the
instruction specifies them. Omitted sizes are inferred from the view, not guessed from
allocation size. ND DMA loop strides are element strides; padding has its own side extents,
constant/nearest selection and fence controls.

`matmul(dst, a, b, *, m=None, n=None, k=None, is_init=True, splitn=None, splitk=None, bias=None)`
means `a @ b.T` for ordinary L1 operands, writing L0C. Explicit `mmad` consumes L0A/L0B.
The MX forms additionally consume scale planes. `conv2d` and `img2col` preserve explicit
geometry and layout requirements. Basic FP16 matmul is independently checked on all four
profiles by [one shared source](../../examples/api/cube_matmul); bias/quant/MX/convolution
teaching cases and cube/vector bridges remain separate pending work.

## A5 registers, masks and SIMT

`Reg(dtype, name="", reg_num=1)` allocates a VF register. `reg_num=2` is restricted to
64-bit or complex groups. `MaskReg(dtype, init_mode=None, name="", *, reg_num=1)` includes
an explicit initial-mask argument; it is not Reg's positional signature. `RegList(dtype,
length, name="")` has a static positive length and statically indexed registers.

Register expressions are materialized by `<<=`. Arithmetic, comparison, method and explicit
instruction forms are enumerated separately. A comparison expression must be assigned to a
MaskReg; `mask * expression` selects active lanes, while multiplying a plain register by a
mask is rejected. Register-list operations expand lane-wise and supported reductions fold a
tree across the registers. A declared register width does not promise that every output lane
of every reduction/conversion is defined.

Explicit VF operations use positional destination/source operands; optional masks may use
keywords where declared. **What a mask does to the lanes it turns OFF is per operator and not
uniform** — most arithmetic writes zero there, the `mask_*` family leaves the destination bit
alone, and a masked store does not write that lane at all, so the memory keeps its bytes. The
per-operator table is [mask write semantics](mask-write-semantics.md); guessing costs a wrong
number in whichever direction was assumed. `cast(dst, src, config=None, mask=None, *, cfg=None)` preserves
the `cfg` alias. `mulscast`/`expsub` accept a trailing mask and keyword-only `layout`.
`ub_to_reg_*`, `reg_to_ub_*`, unaligned-register/cursor operations, interleave, gather/scatter,
mask conversion and packing must preserve their physical lane footprints. Unsupported
width/distribution combinations raise a located gap. The first
[arithmetic example](../../examples/api/axpb) observes every lane of one register.

SIMT functions contain per-thread scalar arithmetic, tensor element access, `cvt`, thread/core
indices, barriers/fences, scalar math/classification and `simt_atomic_*`. Atomic calls return
the old value; compare-and-swap takes target, compare, value. Arithmetic and fence semantics
alone do not prove a concurrent algorithm. The allocated atomic/math samples are still
pending this foundation batch.

## A2/A3 tensor-vector forms

The A2/A3 `add`, `cast`, `compare`, `select`, reductions, gather and related names consume UB
tensors at kernel level. Block and repeat strides are 32-byte blocks. A repeat spans eight
blocks. `count=` and `count_per_rep=` are mutually exclusive and restore the documented
mask state after the instruction. The full stride overloads are explicit in `dsl_vec.pyi`.

`compare` writes a packed predicate into an int8/uint8 tensor. `compare_scalar`'s accepted
keyword is `src2` in the implementation; scalar and source must share the float/integer family.
`SelectMode.TENSOR_TENSOR` needs a distinct destination and an explicit uint32
`tmp_addr_buf` with at least eight lanes. A2 gather/scatter offsets are uint32 byte offsets;
counted gather is rejected because it disagrees with c220 hardware. The
[shared A2/A3 example](../../examples/api/a2_vectors) validates a 70-element counted tail
and compare/select with independent Torch expressions. It does not establish A3 board evidence.

## Host codecs and located gaps

The facade re-exports the intended host codec functions, constants and aliases. Importing
them requires only Python; calling a codec loads its tensor implementation and requires
`ascriptor[torch]`. Their original signatures/defaults are retained.

The [host example](../../examples/api/host_codecs) checks all 256 carriers of both FP4
formats and the legacy E8M0 helper, signed-int4 full/tail carriers, and six HiFloat8 specials.
The legacy E8M0 host helper decodes byte 255 to positive infinity; do not substitute this
for a hardware MX-format NaN rule. FP4 decoding preserves signed-zero bits; encoding an
input negative zero canonicalizes it to positive zero. Negative values that round down to
zero use the negative payload. HiFloat8 ordinary rounding and device cast saturation need
their allocated feature examples; six special values are not full-format coverage.

`zero_mxfp8_l1_padding` is a retained name with no frontend lowering rule. It raises a located
unsupported-operation error; no accepted operative signature is claimed beyond this marker
gap. Use an explicit supported data-movement/fill sequence with a complete reference.
`reset_cache()` is a compatibility no-op. `kernel_print`, `sim_print`, dumps and `print_reg`
are diagnostic operations; emitted debug comments do not claim device-side observability.

The narrow advanced backend protocol is defined by `Backend`, `Artifacts`, `Capabilities`
and `ResourceLimits` in `ascriptor.backends.base`, plus the `ascriptor.backends` entry-point
group. Extensions consume versioned Lowered IR and return relative artifact names and bytes.
The plugin-registration teaching example remains pending; no private pass API is promised.
