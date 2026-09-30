# A5 backend verification and upstream gaps

This maintenance task distinguishes three hardware backends from three
verification stages. The release's 918 reference/sim/pipesim keys did not
establish 918 CCE/PTO/PyPTO hardware results. The canonical backend matrix
contains 306 cases across 30 units, with one board key for each backend.
Each key preserves the unit's independent references and comparison rules.

`GetValueFrom` / `SetValueTo` targeting UB are supported by both upstreams.
The missing paths were in Ascriptor's adapters and are repaired in
M10-061. The
[scalar memory example](../examples/api/scalar_memory) includes
the method and operator spellings; both cases pass A5 CCE, PTO ISA and
PyPTO Pro board checks using the same FP32 inputs and exact comparison.

The [scalar abs example](../examples/api/scalar_abs) covers
dynamic i8/i16/i32/i64/f32 and FP16 widening/narrowing. All six cases pass
CCE and PTO ISA hardware checks, including the sign of zero. Dynamic
FP32/FP16 sqrt also passes both C++ backends. See
M10-062.

The [unified upstream report](upstream.md) owns all upstream issue IDs,
affected forms, evidence, historical performance findings and available
workarounds. A5-UP-001 through A5-UP-008 retain their existing identities;
the prototype and survey findings are reconciled there. Ordinary scalar abs
is A5-UP-001; VF register abs remains a separate supported mapping.

The inspected PyPTO Python files match source revision
`86ef830cb68110cf4092420a4d20c2a664677530` byte for byte (65 files); the
installed version marker alone is insufficient. The validation record also
fingerprints the loaded compiler libraries and CANN/PTO headers. Upstream
gap counts use actual case/backend rows, so a missing form outside the
canonical cases does not inflate the matrix's missing-case count.

The [adapter contract](rfc/0011-pto-isa-backend.md) retains the repaired
scalar-binding, storage-capacity and view rules (M10-063); the
dependency receipt identifies the tested upstreams. M10-064
records the packed-store model correction with independent CCE/PyPTO
hardware observations. M10-075 connects the
predicate spill/fill pair and declares six further printed ops; no canonical case
exercises any of them, which is why a stale declaration and a missing mapping both
survived this matrix. An adapter TODO remains `implementation_gap`;
numerical/compiler failures remain `failed`. Only a located, evidenced
upstream limitation receives `upstream_gap`.

The older blanket refusal of scaled L0C column windows is superseded:
full-height NZ strips have an address-alias representation, including
rotating slots. This connects the existing scaled TMOV path. The independent
FP32 probe then exposed A5-UP-008; this narrower selector defect remains
distinct from the repaired window addressing.

The completed canonical matrix records all 918 keys: cce: 305 pass, 0 upstream gap, 1 failed; pto_isa: 305 pass, 0 upstream gap, 1 failed; pypto_pro: 262 pass, 43 upstream gap, 1 failed. M10-059 accounts for the three retained failures.
That matrix was the kernel repository's `docs/a5-backends.md`. It was removed on 2026-09-23 when
that repository became a gallery of runnable demos, and resolves at kernels
`f79b44b721eba5080b802409c2388f74638405ce:docs/a5-backends.md`. Those counts are its 2026-09-07
outcome for the sources of that day; the gallery keeps no matrix and regenerates none, so they are
dated provenance rather than a current column.
There are no unrecorded case/backend keys or unresolved adapter refusals in
this matrix. The dependency fingerprints
and scalar receipts
retain the actual executed source scope. Missing forms outside canonical
cases, including ordinary scalar abs/sqrt/cast, are listed without inflating
the canonical upstream-gap count.

## Remaining adapter boundaries

The M10-075 predicate spill/fill repair did not qualify runtime-affine UB windows.
The recorded dense sparse-attention route stopped at `mem.slice` when its row
origin depended on `for slot in range(held)`, with `held = Min(tq, s1 - first)`.
Although at most `tq` addresses are needed, the adapter lacked a static bound for
enumerating that TileGroup. This remains a separately recorded implementation
limitation; the mask spill regression
does not establish its repair or a new PyPTO board qualification. A repair needs
a proven enumeration bound and native validation of that window domain.

The accepted [static local capacity contract](rfc/0013-pypto-native-synchronization.md#static-local-capacity)
separately refuses unbounded typed L0 capacity and runtime FIX source pitch
(formerly M10-072). Such refused extensions are not pending implementations.
The guard that enforces it was for a fortnight broader than the contract: it read
a two-sided interval where a capacity needs only an upper bound, so one unbounded
operand erased the constant the other stated, and it refused a `min(constant, ...)`
staging extent whose bound the contract admits. 45 cases in nine attention units
stopped emitting for pypto_pro — 33 to that widening and 12 to the contract
itself. The guard now takes the one-sided bound the contract section above
describes. This source snapshot includes no device receipt for those 33 cases;
rerun them on the target machine before claiming hardware acceptance;
the investigation is M10-100, closed and recoverable through
closed defects. The 12 that remain, all of
`attention/a5_pfa_hif8_task_metadata`, read their extent from a metadata tensor at
run time and are the contract's own refusal; their board column stays a declared
gap, and the canonical matrix above keeps its 2026-09-07 outcome as the historical
result it is.
Pitched FIX right-column windows also retain the [PTO-ISA capacity refusal](../examples/api/cube_vector_roundtrip/main.py)
and its regression; a successful
PyPTO repair does not qualify that other backend.
