# RFC-0011: PTO ISA backend contracts

Status: accepted A5 backend. This page consolidates the implemented rules from
the original movement design and its subsequent corrections. It changes no
operation, comparison rule or qualification. A2/A3 support remains CCE-only;
see [RFC-0012](0012-product-contracts.md).

The complete former RFC, including its numbered batches, rejected explanations,
header comparisons and measurements, is in the
backend investigation archive.
Old source comments naming a historical subsection such as §7.30 refer to that
snapshot. Current case/backend results belong to the
[qualification index](../a5-backend-coverage.md), not the prototype corpus counts.

## 1. Motivation and scope

`pto_isa` prints PTO tile creation and movement over the same Lowered IR used by
CCE. The backend is manual: physical addresses, slots, synchronization IDs and
ownership come from the IR and its passes. PTO automatic allocation/synchronization
must not replace those decisions.

Complete kernels also need scalar/control-flow code, cube operations, GM-list
descriptors and calls to VF/SIMT bodies. VF and SIMT reuse the CCE printers because
those bodies operate on registers and pointers, not PTO tiles. Movement in this
backend must use a supported PTO instruction/helper or raise a located `PtoIsaGap`;
it must not silently substitute a CCE movement implementation.

The [backend capability declaration](../../ascriptor/backends/pto_isa/__init__.py)
and [emitter](../../ascriptor/backends/pto_isa/emit.py) identify implemented forms.
The declared opcode set is read off the emitter's dispatch table and held equal to the
scoped groups by a test; a refusal is a table
row with its reason, never a handler, so it is not declared as a capability.
An opcode handler does not qualify every operand, shape, dtype or toolchain.
Unmapped forms retain the failing operation and source location.

The scalar, control-flow-leaf, debug, event and local-mutex handlers print the same text
as CCE's and are CCE's: both function printers inherit one
[host layer](../../ascriptor/backends/cce/host.py). What stays per backend is how text
reaches the buffer (CCE folds temporaries into their use, this backend prints one statement
per operation), the refusal class, and every handler whose text differs.

## 2. Geometry and specialization

The bridge from DMA burst descriptors to PTO is semantic and uses element units:
fold the view chain into its root, dimensions, offsets, strides and slot, then
construct the matching `GlobalTensor` and tile views. Reusing formatted PyPTO
source strings would lose this geometry. Specialization and affine cancellation
may establish static extents; uncertain bounds or truncated pitch arithmetic
must not be guessed.

PTO separates compile-time physical capacity (`Rows`, `Cols`) from the valid
region. Runtime valid extents are permitted where the instruction supports them;
they do not make allocation capacity or a template row pitch dynamic. Preserve
static `GlobalTensor` dimensions individually, including leading ones required
by ND2NZ. Constructor template arguments must match the declared Shape/Stride
types rather than allowing deduction to turn every dimension dynamic.

Scalar parameters remain runtime parameters unless needed to specialize a static
capacity. They arrive through the shared CCE launch/tiling ABI. Host scalar folding
and emitted arithmetic must follow the scalar IR contract, including its signed
division/remainder boundaries; a backend cannot invent a different interpretation
to obtain a tile dimension.

## 3. Tile creation and physical storage

The [type mapping](../../ascriptor/backends/pto_isa/types.py) owns element spellings,
fractal sizes and layout pairs. Complex memory types have no PTO element type and
retain a located refusal; an integer carrier is not implicit complex support.

| IR memory | PTO representation |
| --- | --- |
| UB | `TileType::Vec` |
| L1 | `TileType::Mat` |
| L0A / L0B | `TileType::Left` / `TileType::Right`, with their distinct fractal orders |
| L0C | `TileType::Acc` |
| Bias table | `TileType::Bias` |
| L0A/L0B MX scale planes | `ScaleLeft` / `ScaleRight` |
| GM | `GlobalTensor` with instruction-specific shape, stride and layout |

`mem.alloc` declares a tile and assigns its pass-owned address. Views, slices,
reshapes and reinterpretations carry geometry to consumers; they do not move
bytes. Cast a GM base pointer to the view's element type before adding an offset
expressed in that type's elements. Workspace offsets and allocation size must be
represented in the host manifest, not merely used in the device body.

A materialized UB tile has an aligned address and a physical row pitch in whole
32-byte blocks. A one-row declaration can round its capacity into padding already
reserved by the allocator while keeping the true valid extent. Widening a
multi-row pitch would change the layout and is refused. A contiguous allocation
that cannot form a legal two-dimensional tile can have a flat pointer carrier;
shape-dependent consumers must build and validate their own view rather than
reuse that carrier as a two-dimensional tile.

Every materialized view, including its alignment padding, must fit within its
parent allocation or one slot. Slot stride uses the allocator's aligned size;
ring indices wrap as `((i % N) + N) % N`. Slot displacement belongs in address
formation, separately from the in-slot offset used for capacity checks. DMA and
VF/SIMT calls must select the same physical version.

The shared Right-window constraint remains relevant to the PyPTO adapter:
native L0B coordinates are `[K, N]` while IR declarations/slices use `[N, K]`.
Static/runtime slots and reinterpretations must swap axes exactly once. For
byte-addressable elements, a whole-N strip at a C0-aligned K origin starts at
`K0 * align16(N_parent) * element_bytes`. A partial-N strip is compact only
within one K fractal; its aligned N origin adds `N0 * C0 * element_bytes`.
A partial-N rectangle crossing K fractals retains the parent's N pitch and
cannot become a compact Right alias. Unsupported origins/pitches remain located
refusals; see M10-067.

## 4. Movement and computation

### 4.1 GM and UB

`TLOAD`/`TSTORE` preserve the actual transfer's valid rows/columns and GM pitch.
The UB pitch belongs to the tile's physical `Cols`, not its valid width. A single
burst does not apply a row gap; a multi-row pitch must remain statically expressible.
Recovering pitch from allocation geometry requires proof that the lowered gap
arithmetic describes that same pitch. An over-declared view is not a valid way
to make a transfer compile.

Atomic Vec-to-GM stores support the declared `AtomicAdd` form; max/min remain
explicit gaps. A per-element-strided GM gather has no general tile-load spelling.
Vec-to-Vec `TMOV` can use a register loop rather than the CCE DMA engine, so matching
results do not establish identical instruction cost.

### 4.2 GM to L1

ND and DN sources use their respective `GlobalTensor` layouts; the source pitch
occupies stride 3 for ND and stride 4 for DN. Mat physical capacity preserves
the NZ column-block height while valid extents describe the transfer. Byte copies
and convolution feature maps retain their own layout/footprint contracts.

### 4.3 L1 to L0 and the bias table

Use `TEXTRACT` for an L0 operand window into a larger L1 tile. `TMOV`'s equal-shape
assertion cannot express that window without changing the source's NZ pitch.
L0A and L0B use opposite fractal-order pairs. A transposed operand selects the
appropriate source fractal view; it does not transpose storage in place.

Operand capacity comes from the physical IR allocation, with K last on both IR
operands; transfer/MMAD extents bound the valid region. Changing capacity between
the load and the matmul would change the fractal address map. Bias transfers use
the dedicated Bias tile and retain the admitted dtype and footprint checks.

### 4.4 L0C output

An accumulator's physical `Rows` carries `align16(M_src)` as source pitch;
valid rows/columns carry the actual M/N transferred. `CompactMode::Null` preserves
that physical pitch for stores; it must not be inferred from whether valid extents
are dynamic. Compact mode is instruction-specific.

| Destination | Representation and boundary |
| --- | --- |
| ND GM | `TSTORE` into the matching logical matrix and row pitch |
| NZ GM | The destination plane is `[1, N/C0, M_pad/16, 16, C0]`; its height determines destination stride, separately from transfer M |
| Transposed GM | Use the NCHW store arm with shape `[1, 1, N, 1, M]` and destination pitch in stride 2; `Layout::DN` itself has no accumulator-store arm |
| L1 | Windowed Acc-to-Mat transfer, as specified in §4.13 |
| UB | Explicit Acc-to-Vec mode; see §4.11 |

NZ stores require a proven aligned plane and an admitted dtype/C0 combination.
The historical partial-plane mapping distinguishes PTO's debug assertion about
valid rows from the instruction's physical stride; it is not permission to discard
geometry checks or admit an unmeasured byte-output format. I039
retains the distinct byte-stride boundary.

### 4.5 UB to L1

Use `TMOV` for supported contiguous flat Vec-to-Mat transfers, `TINSERT` for NZ
publication, and a column-wise `TINSERT` sequence for ND-to-NZ. A gapped flat
transfer cannot be silently collapsed to a contiguous one. Same-width element
stand-ins are allowed only for bit-preserving transfers with the required geometry.

### 4.6 Memory descriptors and GM lists

`mem.get_buf` selects a physical slot without changing its layout. `GMList`
parameters use the shared descriptor protocol and resolve to GM pointers;
descriptor access is not an additional tile instruction or a new launch ABI.

### 4.7 Cube multiplication

MMAD operand shapes and accumulator window attributes must agree. Initialization,
accumulation and bias select their corresponding matmul forms. Do not resize an
L0 operand to an unaligned valid K merely because a particular matmul consumes
fewer elements than the allocation holds.

### 4.8 NZ publication

For UB-to-L1 NZ insertion, source `Rows` is the physical source column-block
height, valid rows/columns describe the transfer, and destination `Rows` is
`align16(m_dst)`. Keep the source's `CompactMode::Null`: substituting a rounded
valid row would change its source gap. Header bodies, not only signatures, must
be checked against the actual vendor installation; old TINSERT mode names and
gap formulas are not interchangeable across the inspected header versions.

### 4.9 L1 fill and padding

`TEXPANDS` derives the fill size from tile capacity. A fill must cover exactly
that admitted capacity and fit its repeat range; a partial fill is refused.
An unsupported floating element can use a same-width integer view only for a
qualified bit pattern such as zero, not arbitrary numeric substitution. Preserve
the emitted MTE2 barrier after the fill. NZ `TFILLPAD(Mat)` accepts positive-zero
padding only; negative zero and nonzero padding retain explicit refusals.

### 4.10 ND-to-NZ publication

The conversion emits one `TINSERT` per C0 column. Column i starts at `i*C0` in
the ND source and `i*C0*align16(m_dst)` in L1. Each iteration transfers a full
C0-wide block, including the final block's physical footprint. The column count
and pitches must fold; supported valid row counts can remain dynamic. This is
the existing wrapper's decomposition, not hidden scratch allocation or a new
layout-conversion algorithm inside the backend.

### 4.11 Quantization and Acc-to-Vec mode

Scalar quantization preserves the mode and the packed dequantization word:
scale bits, offset and signed saturation follow the actual dtype pair. Only
the appropriate eight-bit arms carry an offset and only signed eight-bit output
sets its saturation bit. An absent scale differs from an explicit zero scale;
unsupported pairs and riders must refuse rather than fall through to `NoQuant`.

HiF8 hybrid applies only to the admitted FP32-to-HiF8 ND GM store with explicit
scalar scale. That path names `QF322HIF8_PRE_HYBRID` while retaining PTO geometry
checks. Other stores must not accept the flag and silently use ordinary rounding;
see M10-027.

Acc-to-Vec scalar quantization and downcasts require SINGLE with the chosen vector
subblock. SPLITM/SPLITN carry the same-type plain copy only; they do not support
relu or non-default requantization. The adapter explicitly selects PTO's existing
`TMovCcToUb` quantization template for supported FP32/FP8 cases where an ordinary
dtype selector would choose the wrong mode. Unsupported relu, atomic or clipped
relu riders on an L0C output remain located gaps.

Static partial-M Acc-to-Vec keeps `Rows=M_src, ValidRow=M`, with every physical
source/destination footprint within its parent. No vendor header is patched.
The public [Cube API](../api/cube.md#draining-l0c-into-the-vector-side) owns the
author-facing mode restrictions.

### 4.12 Microscaling and packed operands

An MX operand uses separate `TEXTRACT` operations for data and scale. The scale
plane address is the data address divided by 16. Its shape divides the data's
fractal-inner axis by C0, and Left/Right scale layouts are distinct from the data
layouts. GM scale loading preserves the e8m0 half-lane packing and the required
trailing dimension of two. Reconstruct the same scale tiles at matmul so types
check the same physical planes. Accumulation uses the overload naming the
accumulator as both input and output, not a nonexistent `TMATMUL_MX_ACC`.

Packed FP4 dimensions count logical elements, while byte carriers count packed
storage. The low-level load and composite desugaring must use the same units.
PTO's KHALF-aware load count is the qualified one-fractal count for a 16x64 FP4
operand; an extra unused CCE load from an old implementation is not a requirement
to reproduce an over-read.

### 4.13 Acc-to-Mat and convolution

Windowed Acc-to-Mat uses `TEXTRACT` so the valid M survives; `TMOV` would replace
it with physical source Rows. The retained plain window form requires a two-byte
destination. The separately admitted whole-row FP32 path uses `TINSERT`;
partial FP32 rows must not inherit that qualification. Preserve relu on the
supported form and its independent control.

`TIMG2COL` uses a `ConvTile` with the declared feature-map/window/padding fields,
one C0 chunk per instruction and the actual valid M. Its automatic SPR writes
must carry the same values even if they repeat work the CCE wrapper hoists.
An M position beyond `align16(Ho*Wo)` can leave the physical plane; it is not
ordinary padding that a larger tolerance fixes. Comparison of computed padding
belongs to the canonical kernel's current contract, not a prototype mask table.

## 5. Synchronization, VF and SIMT

PTO automatic mode is excluded. Pass-assigned event IDs/pipes and slot versions
remain explicit. The support `Flags`/`Event` types preserve preset tokens,
depth-dependent rotation and scope-exit drain, including early returns; the
superseded static drain analysis is not the current protocol. Deep events must
name the support type explicitly to avoid collision with PTO's own `Event`.

Cross-side ready/free helpers preserve both vector participants: the AIC side
sets/waits N and N+16 where the AIV flag remapping requires it. Mutex capacity
tokens start on the consumer side and drain on the producer side. Global-group
channels remain distinct from local point-to-point flags. Local mutex operations
emit the IR-selected get/release operations; the backend allocates no extra IDs.
Ordering required by lowered synchronization or a qualified instruction mapping
must not be removed because operations appear on the same pipe.

Manual Vec tiles expose typed UB pointers via `.data()`; VF calls use those
pointers, casts and offsets. `__cce_get_tile_ptr` is not the manual call bridge.
Reuse the CCE `VfPrinter` and `SimtPrinter`, preserving grouped registers and
located refusals. SIMT launches carry their actual thread count; collect launch
sites before rendering launch bounds and retain the largest count for a shared
callee. Floating atomic accumulation remains order-sensitive under its contract.

The support header owns scalar helpers, events, slot wrapping, GM-list and SIMT
bridges; it does not hide PTO movement behind a second wrapper API. It stays a separate
file from the CCE header, and every definition it restates from that header is the same text:
a test compares them, after the SIMT rounders and
`fmod` had fallen two CCE repairs behind and rounded differently on the two backends. Per-side entry
functions use the shared CCE launch ABI, task kind, scalar tiling fields and
workspace manifest. A refusal must restore printer state and emitted-line buffers
so partial output cannot be counted as a successful translation.

## 6. Refusals and support boundaries

Keep unsupported complex memory, per-element-strided tile loads, unrepresentable
capacity/pitch, unsupported quantization riders and instruction geometry explicit.
The current [upstream report](../upstream.md) distinguishes a missing vendor form
from an adapter defect; the [backend qualification](../a5-backend-coverage.md)
separates emission, compilation and device outcomes. Do not revive the old claims
that SIMT, FP4, transposed GM stores or every HiF8 hybrid form are blanket gaps.

## 7. Validation and historical batches

The current backend tests cover capacities,
dynamic extents, slot/address agreement, event presets/drains, the VF/SIMT bridge,
movement/quantization forms and invalid-call refusals. They do not execute vendor
hardware. This first source snapshot includes no historical device receipts, so
what a kernel does on this backend today is answered by running its demo —
`python main.py --launcher board --backend pto_isa` in the demo folder, with a
[CCE control](#8-dependency-and-measurement-discipline) on the same case before a failure is
called a port defect.

The old §7.1–§7.33 acquisitions are restored through
history. They use their original corpus,
case inputs, compiler and source identities. A CCE control is additional diagnostic
evidence, not a replacement for the independent generated reference required by
[RFC-0003](0003-functional-goldens.md).

## 8. Dependency and measurement discipline

Inspect the headers and compiler actually used for the selected target, including
template bodies and assertions. Equal version strings or declarations do not
establish equal instruction expansions. Source emission, vendor compilation and
device comparison are separate gates; a successful corpus run does not cover a
branch or dtype absent from that corpus.

Preserve failures, unstarted jobs and control identities. Do not treat a transfer
failure as a kernel verdict or an unwritten/NaN result as correctness. Input
generation, complete-output checks and tolerances belong to the current unit
contract. Neither this consolidation nor historical recovery requalifies current
source, hardware, performance or a wider device family.
