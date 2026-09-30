# RFC-0013: PyPTO-Pro emission of IR-owned mutex IDs

Status: development contract, 2026-09-15. This replaces the experimental
backend synchronization planner at the user's request. The IR owns allocation
and synchronization; the backend translates its buffer identities.

## Selection and migration

`emit_module`, `PyptoProBackend.compile`, `compile_kernel` and
`OpExec(..., launcher="pypto")` default to `sync_mode="manual"`: `@pl.jit(auto_mutex=False)`
with the IR's own `sync.local_mutex_get/release` printed as
`pl.system.mutex_lock/mutex_unlock`, so this backend owns the credits exactly as the cce
backend does with `get_buf`/`rls_buf`. Explicit `sync_mode="auto_mutex"` emits
`@pl.jit(auto_mutex=True)` and delegates the credits to PyPTO; it is retained for comparison
and is **not** sound for every program (see "Delegated credits are per Tile group" below).
Other launchers keep their existing behavior and reject PyPTO synchronization options.

This default was `auto_mutex` until 2026-09-22 and was changed on board evidence (D-260).

The former event selector, ACK closure, fill exclusions and physical-component
ID allocator are removed. The experimental `keep_events` option is removed;
callers must omit it. Existing typed events, raw flags, barriers and cross-core
protocols are emitted as specified by the IR. The backend neither removes nor
invents their synchronization. No A5 event-budget throttle is restored.

The adjacent all-vector ready/wait pair emits `pl.system.sync_all` with
`core_type=pl.SyncCoreType.AIV_ONLY`. The current
[PyPTO API](https://pypto.gitcode.com/api/pro_api/SIMD-API/synchronization/sync_all.html)
uses FFTS hardware synchronization and accepts only the `core_type` keyword;
it does not expose a `mode` argument or a `SyncAllMode` enum. Do not emit the
older `mode=pl.SyncAllMode.HARD` spelling. Omitting it preserves the hardware
barrier, participant set and ordering; it does not select a software fallback.

## Exact identity mapping

Each `mem.alloc.mutex_ids[i]` is the mutex ID of physical slot `i`.
The corresponding `pl.make_tile_group` member receives exactly that ID, without
renumbering. This includes default L0A/L0B scratch allocations and their typed
views: all IDs already exist in IR, with no backend reservation or allocator.
A managed Tensor becomes a one-member group and a `[0]` binding.
Addresses, slot pitch, reserved capacity and side remain unchanged. Short UB
rows may use the padding already reserved by address allocation. Each AIC and
each AIV has its own namespace of IDs 0 through 31, shared across banks.

Backend-created aliases inherit the IDs of the physical slots they overlap.
A wide alias may carry multiple existing IDs using PyPTO's per-Tile ID list;
it never merges or renumbers the underlying slots. A carrier whose type is wider
than its proven access window inherits that window's parent-slot ID instead of
treating the full carrier type as an access. This includes ND-to-NZ UB source
columns: their padded carrier rows preserve the native layout, while the
valid row count and column width bound the actual read to its parent slot.
All members of a PyPTO
group must have the same nonzero number of IDs. Unsupported mixed groups,
unresolved geometry or changed native capacity produce located diagnostics.
Unmanaged buffers receive no invented IDs. Caller IR remains unchanged.

In native mode `sync.local_mutex_get/release` emit no Python statements or
comments. PyPTO inserts actual local locks around accesses to the managed
Tiles. The backend must verify those guards
resolve to managed allocations; it must not silently discard an unbound lock.

Unsupported upstream operations on managed Tiles fail explicitly rather than
silently falling back to the removed event planner. Managed Cube scalar fill
emits `pl.expands(Mat, value)` with the original IR IDs and the existing
post-fill MTE2 barrier. Upstream's memory-aware pipe selection applies
([A5-UP-038](../upstream.md#a5-up-038), fixed in `6a652e733`): Mat selects MTE2, Vec selects V.
Whole-tile capacity and immediate-value checks remain in force. The separate
padding-only fill this sentence used to exclude, `dma.fillpad_l1`, was retired on
2026-09-22 (library `f75ee23`): a Cube scalar fill is now always the whole tile.

Omitting these operations does not delete shared IR synchronization. Original
operations remain inspectable in IR; allocation IDs remain in Tile declarations
and artifact metadata in **both** modes. Empty emitted sections retain a `pass` so the Python
source remains valid. Explicit manual mode still emits local mutex calls.

### Delegated credits are per Tile group, not per byte range

PyPTO's inserted locks order accesses to **one Tile group**. They do not order an access
through one group against an access through another, even when both groups carry the same
mutex IDs and cover the same bytes. Native mode is therefore sound only for a program in which
every managed access to a byte range goes through a single group.

This printer does not always produce such a program: it mints an alias group (the `_st` suffix)
for a matmul destination, so a cube matmul writes through `X_st` while the FIX-pipe copy of that
same L0C reads through `X_grp`. Under `auto_mutex` nothing orders that pair, and the copy takes
whatever the previous writer left in L0C — the previous launch's accumulator, since on-chip
storage is not cleared between launches. The measurement that established this, on 2026-09-22:
the same kernel and the same inputs are exact alone and wrong by 0.0025634765625 after a
predecessor, bit-reproducibly across three runs, and exact again once the edge is supplied. Both
units it corrupted, `delta_rule_bwd` and `gdn_bwd`, then passed through the normal unit path on
hardware with the repair, and in a 33-unit board sweep of the new default they were the only two
units whose outcome moved at all. The investigation record is D-260, closed and recoverable
through closed defects; its regression guard is
`tests/backends/test_pypto_alias_group_credits.py`, which asserts on the emitted text that one
mutex id is held on PIPE M across a matmul and retaken on PIPE FIX before the copy that reads its
destination, and that `auto_mutex` refuses at the alias it cannot order.

A backend that delegates therefore **must** refuse at the point it creates an alias group over
managed bytes, rather than emit a program whose credits it cannot account for. Manual mode has
no such condition: the IR's own mutex operations are printed and ordered by ID, so a producer
and a consumer on different groups over one allocation are ordered by that allocation's ID —
which is what the cce backend has always done with `get_buf`/`rls_buf`.

## Reloaded L0 storage and narrow Bias aliases

An origin reload into a larger L0 allocation establishes the compact layout
described by that DMA, independently of the allocation's maximum dimensions.
For supported static ordinary operands, backend lowering materializes a typed
view for the load and its reaching MMAD consumers. The allocation, physical
slot pitch and mutex IDs remain unchanged. Track the reaching load through
structured control flow; refuse ambiguous layouts instead of reusing a view
from an arbitrary branch or loop iteration. A slice of an already loaded
matrix remains a parent-pitch window and must pass the existing byte-map proof.

A narrow Bias Tile is an alias of its original BT slot. Register its physical
bank, address, width and dtype so that it inherits the original IR mutex ID.
Rotating aliases retain the full parent slot pitch and captured slot selection;
their narrower width does not change storage spacing or allocate new IDs.

The same rule applies to flat GM-loaded MX scale blocks in L1 and the UB
sub-tiles used to decompose a static higher-rank DMA. Each alias must remain
inside its parent slot, inherit its physical mutex identity, and preserve the
original slot pitch and selected index. Neither adapter creates scratch storage.

## Sorting subset

`vec.mergesort4` is deliberately unsupported in the PyPTO-Pro backend in both
`auto_mutex` and `manual` modes. Reject it at its source location before any
multi-pass merge expansion. The former expansion allocated temporary UB storage
outside IR; that expansion and its scratch allocator are removed at the
maintainer's request. `vec.sort32` and the equal-length, physically adjacent
`vec.mergesort_2seq` mapping to one `pl.mrgsort` pass remain unchanged. This
restriction leaves the DSL operation and other backends unchanged.

## Static local capacity

PyPTO local Tile allocation dimensions, slot counts and addresses must resolve
to compile-time integers. Runtime-sized local capacity is unsupported and must
produce a source-located refusal. A runtime valid extent is a different property:
it may select part of a fixed allocation if its typed physical capacity has a
proven constant bound. That bound is **one-sided**: a capacity asks only how
large the extent can get, so an operand that states an upper bound is enough even
when the other states nothing. `min(a, b)` is at most the smallest upper bound
among the operands that have one; `max(a, b)` needs both; and `x & m` with a
non-negative constant mask is at most `m`, because two's complement clears every
bit `m` does not set. A two-sided interval must not be required here, and it was:
the guard read one, so a single unbounded operand erased the constant the other
stated outright, and refused 33 of 45 attention cases whose extent was
`min(128, ...)`. Do not use raw byte-carrier axes as typed bounds, grow a
Tile from runtime metadata or clamp the requested work to make it fit. Dynamic
FIX source pitch also remains unsupported. Authors can specialize dimensions or
use static branches with correctly sized carriers. This policy does not remove
dynamic-capacity support from the shared IR or other backends.

## Generated VF parameters

After rendering a VF body, remove formal parameters with no name reads in that
body and the matching positional arguments at every call. Include generated
byte carriers and exact-constant parameters in the same ordered signature.
Inspect nested control flow; comments and similar names are not references.
Do not merge used parameters by address, alias or mutex ID. The public kernel
ABI and caller IR remain unchanged.

HiF8 carrier adaptation can leave an unused original Tile parameter beside its
used UINT8 view. Removing the former avoids passing a redundant mutex owner to
native auto_mutex. Construct the carrier from the original call operand before
pruning positions, preserving runtime slot selection. This is an Ascriptor
interface cleanup, not a repair of upstream mutex deduplication for used aliases.

## Integer scalar width at VF division

The integer scalar `pl.cast` compatibility supplement is required when emitting
ordinary integer `scalar.cast` or restoring narrow VF quotient/remainder
operands. Map an explicit integer cast to `pl.cast(value, result_dtype)`; retain
located upstream gaps for float, bool and sub-byte conversions.

For a VF integer `scalar.div` or `scalar.mod` whose result and value operands are
ordinary integers no wider than 32 bits, preserve each operand's IR dtype with
`pl.cast` and give untyped integer literals the result dtype through `pl.const`.
Literals must fit that dtype. Do not narrow a genuine 64-bit value or change a
floating expression. Native integer-division expansion continues to own signed
floor/truncation correction, modulo, ceiling and alignment semantics. This is
typed source emission, not an IR arithmetic rewrite or a VF scheduling change.
The caller module, memory allocations, mutex IDs and VF invocation structure
remain unchanged. [Dependency recovery](../pypto-pro-supplements.md) owns the
required PyPTO front-end supplement.

## Validation

MX scale companions inherit the corresponding L0A/L0B data-slot mutex ID.
Their hardware address is the data address shifted right by four; the adapter
must prove the bank pairing, address relation and capacity for every slot.
They allocate no new IDs. Rotating L1 scale staging and its flat byte-copy
aliases retain the original L1 slot IDs and selection index.

Regression tests cover all public defaults, explicit manual comparison,
nonzero/reordered IDs, singleton and rotating buffers, physical aliases,
independent sides, range and geometry errors, immutable caller IR, preserved
events/barriers/cross-core protocols and removed option diagnostics. Source and
installed-wheel checks are separate from upstream compilation and hardware
execution. Device qualification names the exact generated workload and vendor
revision; enabling native mutexes is not a proof of arbitrary GM ordering or
same-pipe alias safety.
