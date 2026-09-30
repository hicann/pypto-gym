# Storage positions and views

Use the `Position` namespace from the selected device facade. Its values are
immutable compiler tokens. They are not tensors, allocation handles or C++
objects; `PositionType` and a public `.cpp` property are not successor APIs.
`repr(Position.L1)` is `Position.l1`, and comparison with the string `"L1"`
returns false. Pass the namespace member to the DSL.

| Position | Role | Minimal use |
|---|---|---|
| `Position.L1` | Cube operand staging | [One cube tile](../../examples/api/cube_matmul) |
| `Position.L0A`, `Position.L0B` | Explicit cube operand storage | Staged `l1_to_l0` followed by `mmad`; the operand layout must match the consumer |
| `Position.L0C` | Cube accumulator | [A5 cube/vector ownership](../../examples/api/cube_vector_roundtrip) |
| `Position.UB` | Vector and VF local storage | [A5 register groups](../../examples/api/register_groups) |
| `Position.BT` | Cube bias table | Bias belongs to the initialization step; capacity and conversion depend on the device/backend |
| `Position.GM` | Global-memory metadata value | Use typed `GM[...]` arguments or `GMBuff`, rather than a local `Tensor` allocation |

Local `Tensor`, `DBuff`, `TBuff` and `QBuff` shapes are two-dimensional. A
buffer's slot count and the events or mutexes governing its lifetime are
separate obligations. More synchronization credits do not create more storage.
Conversely, a kernel may retain more physical slots than it wants to keep in
flight. `DBuff`, `TBuff` and `QBuff` accept a keyword-only `sync_depth=N` in
`1..slots`. It preserves the allocation size and modulo indexing, but sets the
credit count of the A2/A3 slot session guarding that allocation: its ledger
declares that depth in both directions, so the producer keeps N windows in
flight and the reverse credits are what stop it there. This is a scheduling
escape hatch, not an aliasing waiver: explicit events and mutexes are unchanged,
and `guards=buffer` still derives mutex credits from the physical slot count.
On A5, the implicit local mutex strategy assigns one ID per physical slot and
does not apply the `sync_depth` cap.
The [two-slot ring](../../examples/api/buffer_ring) wraps five iterations;
the [A2/A3 bridge](../../examples/api/cube_vector_bridge) gives both vector
readers their own rows and returns a GM slot after both MTE2 reads complete.

`.T` records transpose intent for a consuming operation. The frontend accepts
it on L1/L0A/L0B/L0C and suitable two-dimensional GM views; that does not imply
that every DMA or cube operand can consume every form on every backend. UB
transpose is rejected. The cube example validates its L1 operand convention;
use the applicable operation's shape/layout contract for other consumers.

GM view strides and offsets are in elements of the current dtype. A view does
not copy or enlarge its root allocation. `reshape` and `reinterpret` therefore
require compatible physical storage, and an NDDMA destination row must meet
its transfer alignment. The [GM view unit](../../examples/api/gm_views)
checks padded-row overlap, non-unit-stride reads and a rank-three window against
independent slicing. Non-unit-stride writes are rejected by their DMA consumer.
An explicit MTE3 barrier protects its overlapping rewrite.

Device profiles and backend resource limits define capacities. Historical
tables in the former API documentation are source context; the `Position`
token itself does not promise a device-independent byte capacity or transfer.

## Tensor and list authoring

Use `GM[dtype, shape]` and `GMList[dtype, member_shape]` annotations for external
storage. A list has a runtime `count`; each selected member has its declared
shape, including supported ragged dimensions. A returned list reuses the
caller's buffers. [Concatenation](../../examples/api/list_concat) and
[splitting](../../examples/api/list_split) independently observe every
member and its ordering.

The following source-framework forms predate the successor 0.1 contract.
They are not supported compatibility methods or patch removals from that contract.

| Former source form | Current authoring route |
| --- | --- |
| `Tensor.set_shape(...)` | Construct an immutable slice, `view` or compatible `reshape`; retain the root's physical bounds |
| GM `bind_cv_mutex` / `bind_vc_mutex` and forwarded `lock/ready/wait/free` | Create an explicit `CvMutex` / `VcMutex` and call its ownership operations |
| GM-list `size()` / `item_numel(...)` | Use `count`, supported iteration/length and the selected member's shape |
| A2 `tensor.VecOP(...)` builders | Call the explicit [A2/A3 tensor-vector helpers](a2-vectors.md) |
| `source_buf`, `source_index` and allocation-counter mutation | Use declared views and buffer slots; generated names and internal roots are not authoring state |

The located diagnostics in `test_api_operand_boundaries.py` reject the old
mutator/builder forms. `reset_cache` remains a compatibility no-op; it does
not reset generated allocation accounting.

## Indexing and slicing

A subscript names every declared dimension. `x[1, 0:16, :]` reads a rank-three
GM parameter; `x[1, 0:16]` is rejected with `index has 2 dimension(s), tensor
has 3`. Integer indices and slice bounds may be static ints, `Var`s, kernel
scalars or index expressions (`begin : begin + 16`), and an omitted bound is the
view's current extent. Sliced dimensions are kept in order and integer-indexed
ones drop out, so `x[i, a:b, :]` is a rank-two view.

| Form on a GM tensor | Result |
| --- | --- |
| `x[i, a:b, :]` | Accepted; the rank-two view above |
| `x[i, j, k]` | Rejected: `index at least one dimension of a GM tensor with a slice`. One element is `Var.GetValueFrom`, not a subscript |
| `x[a:b, c:d, e:f]` | Rejected: `at most two dimensions of a GM tensor may be sliced` |
| `x[i, a:b:2, :]` | Rejected: `slice steps are not supported` |
| `x.T[a:b, c:d]` | Rejected: `a transposed view cannot be sliced`; slice first, then `.T` |

A rank-three parameter therefore carries an integer index on one dimension —
`x[head, rows, :]`, never three slices — and a loop over that dimension is the
authoring form for a batch or head axis. [Indexed reads](authoring.md#scalar-values-and-memory)
resolve a single element instead, and a subscript read out of memory is how a row
gather is spelled: [indexed row gather](../../examples/api/indexed_row_gather)
copies one row per slot, at each table's own width.

A local `Tensor`, `DBuff`, `TBuff` or `QBuff` has different rules: one integer
subscript is an element offset (`ub[16]`), further integer subscripts are
accepted with no slice at all (`ub[1, 5]`), and a slice keeps every dimension.
The one-slice minimum and the two-slice maximum are GM rules only.

## Base-install padding literals

CCE source generation for typed FP32, FP16 and BF16 DMA padding literals requires
no `torch`, `numpy` or `torch_npu` import. Encoding uses `struct` and integer bit
operations. Narrow formats first round through FP32, preserving the scalar
constructor's signed zeros, finite overflow, subnormal boundaries and ties.
FP32 NaNs keep the representable quieted payload; FP16 NaNs keep their sign with
a canonical payload; BF16 NaNs become positive `0x7fc0`. Integer and opaque-byte
encodings retain their separate behavior.

The padding regression blocks
optional tensor imports and compares generated encodings with independent controls.
The source review and
installed regression
retain their actual artifact identities and historical negative control (M10-056).
Literal emission is separate from running the kernel or its host reference.

## Transfer units and initialized regions

| Transfer | Payload and gap units |
| --- | --- |
| `gm_to_ub_pad` | Payload and GM source gap in elements; UB destination gap in 32-byte blocks |
| `ub_to_gm_pad` | Payload and GM destination gap in elements; UB source gap in 32-byte blocks |
| `ub_to_ub`, raw `ub_to_l1` | Payload and both gaps in 32-byte blocks |
| `gm_to_ub_nd_dma` | Loop extents and strides in elements of the operand dtype |

Raw local bursts copy complete blocks without repacking or filling gaps. A
logical partial block therefore still needs enough allocated source and
destination bytes. [Padded DMA](../../examples/api/dma_padding)
uses a literal nonzero `pad` and checks all 16 FP32 lanes after a nine-element
input burst. [ND transfers](../../examples/api/nd_transfers) check
constant padding, edge replication, row selection, transpose and BF16 padding.
The current NDDMA declaration exposes `fence`; the old `asc_optimize` option
is outside that declaration.

`ub_to_l1_nd2nz` reads an ND source using its physical row pitch; `ub_to_l1_nz`
reads a source already packed in compact NZ order. Destination NZ pitch,
source pitch, slice offsets and visible logical extents are distinct. A copy
does not initialize untouched L1 rows, tail lanes or later blocks. Producers
must initialize every byte later consumed by a cube operation. The
[A5 roundtrip address formula](../../examples/api/cube_vector_roundtrip/main.py)
connects register lanes, compact UB pitch and L1 pitch. Its 64-element FP16/BF16
cases separately observe all staging padding/guard bytes and all 16 executed
cube rows, including the eight explicitly initialized zero rows when M=8 is live.

`transdata5hd` belongs to the shared A2/A3 vocabulary. It performs a 16-by-16
UB bit transpose for matching b16 operands. Row strides use elements; repeat
strides use 32-byte blocks. Its declaration and located operand checks do not
establish numerical or hardware coverage for every dtype/stride combination.
