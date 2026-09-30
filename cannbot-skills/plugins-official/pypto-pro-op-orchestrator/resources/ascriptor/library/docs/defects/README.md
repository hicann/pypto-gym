# Unresolved defects

Updated 2026-09-24. Four implementation/model defect IDs remain open: two M10
records and two importer/model records. Passing kernel workarounds do not close
their underlying generic defects. The table retains the original failure and
closure requirements; historical executions do not qualify the current runtime.

| Defect | Current result | Attempted solution and result | Next solution and closure condition |
| --- | --- | --- | --- |
| [M10-088](M10-088-masked-stride-one-load-footprint.md) — masked load footprint | A5 CCE/PTO stride-one block loads read the full register before masking; the model reads only active addresses. | HiF8 uses a bounded UNPK load and deinterleave. This fixes the kernel, not the generic emitter. | Preserve the active-address footprint in native code; check final valid/first invalid addresses, zero predicates and register carrier types. |
| [M10-040](M10-040-block-quant-model-precision.md) — cube accumulation precision | The installed `matrix_block_quant/pack4_reuse` model still fails the original exact E5M2 payload comparison. Its previously measured board case passes. | New generated FP16 cube cancellation captures at K64/K512 disagree with FP64-end-rounded accumulation and five chunked FP32 hypotheses. An independent exact basis/orientation control passes. These simple replacements cannot establish a silicon model. | Obtain or measure the cube's internal grouping, precision and rounding rules, then qualify them across generated shapes/seeds and all 12 quantization cases. If exact emulation is unavailable, review the numerical contract separately. Keep the exact original failure; do not fit a tolerance to its one byte. |

| Defect | Current result | Mitigation and closure condition |
| --- | --- | --- |
| [I012](I012-simulator-gm-cache-line-stores.md) — cross-core GM scalar stores | The model warns but retains stores that A5 can lose when different cores share a 64-byte cache line. | Use the measured producer clean plus ordered publication protocol, or separate cache lines. Model the lost-store behavior and qualify the positive/negative protocol controls. |
| [I033](I033-simulator-nz-window-offsets.md) — offset NZ UB windows | Model/register/DMA accesses and scalar folding disagree on the origin: the recorded example starts at element 136 versus fractal element 40. Current canonical kernels do not expose this access pattern. | Settle the per-access layout rule, measure a native probe, and align frontend, model and printers. |

The [upstream report](../upstream.md) owns dependency restrictions and their
workarounds. Its dated issue count is not an additional count of current
Ascriptor defects. The [backend boundaries](../a5-backend-coverage.md#remaining-adapter-boundaries)
retain the separately scoped runtime-affine UB window limitation reported with
the repaired M10-075. Unsupported dynamic PyPTO capacity is an explicit
[contract refusal](../rfc/0013-pypto-native-synchronization.md#static-local-capacity),
not an implemented extension. M10-100 was the separate defect that the guard
enforcing it also refused extents the contract admits; it is closed, and the
contract refusal it was confused with is not.

## Closed on 2026-09-24

I044 — the pipe model refused a FIX transfer out of a row-offset L0C slice that fits.
`fix_errors` compared `origin + extent` against the allocation with the two measured in different
layouts: the source extent is always an NZ end with `c0 = 16`, because that is how the FIX
descriptor walks the accumulator, while `Memory.origin` fell back to row-major strides derived from
the logical shape. For `examples/api/cube_quant_ub`'s `split_i8` —
`ub1 <<= l0c[16:32, :].requant(...).subblk(0)` out of a `[32, 64]` FP32 accumulator — that is
4096 + 7168 = 11264 against an 8192 allocation, where the NZ origin is 1024 and the transfer ends at
8192, exactly the allocation. The 3072-byte difference between the two origins was the whole
refusal, and the module's own rule is that a rejection requires a provable overrun.

`Memory.origin` already had an NZ branch, conditioned on `layout == 'nz'`. **Only the model ever
reached the wrong one.** `StaticMemory` takes the layout from the `MemType` and the frontend gives
every L1/L0A/L0B/L0C tensor `nz`, so the static verify path had been computing an NZ origin all
along; `backends/sim/dma_ops.py::_check_fix_bounds` builds its `Memory` from a `MemRef`, whose
`layout` only remembers NZ-packed UB windows (`.nz()`), so an L0C arrived there with no layout at
all. The condition is now keyed on the space — A5 L0C is fractal however the model's logical
row-major tensor is shaped (D-022) — which makes the two construction sites agree without asking a
`MemRef` to carry a layout it was never meant to hold.

**The destination direction carried the same defect**, which the closure condition asked for before
the repair could be called general. `dma.l0c_to_l1` computes its destination extent as an NZ end
with `c0 = 32 / width` unconditionally, so a row-offset L1 destination was refused at a row-major
origin against an NZ footprint: a `[32, 64]` FP32 tile written 16 rows in at its own pitch reports
4096 + 7680 against 8192, where the NZ origin is 512 and the transfer again ends exactly at the
allocation. Keying on the space would be wrong here — an L1 tile may legitimately be ND, which an
L0C may not — so `fix_errors` now **names the coordinate system it measured each extent in** and
`Memory.origin(layout)` reads the origin in that one, rather than either side inferring it from a
declaration the model had already dropped. The source is always `nz`; the destination is `nz` for
`dma.l0c_to_l1` and the memory's own for the rest, which is what the row-major `l0c_to_ub` and
`nz2nd` extents and the linear GM ones already assume. Nothing had exercised it wrongly: the one
measured row-offset L1 destination, the importer's `acc_insert`
(`tests/importers/pro_p6_dma_followup.py`), inserts at row 8 and is admitted at both origins
(512 + 1536 and 256 + 1536, against 2048).

Measured in the models: in `examples/api/cube_quant_ub`, `python main.py` passes 7/7 and
`python main.py --launcher pipesim` now passes 7/7 too, `split_i8` bitwise exact over all 2048
output bytes under each; it was skipped on `pipesim` before. The `REFUSED` entry, its `# pipesim:`
comment and the two `metadata.json` paragraphs that described the refusal are gone, and
`examples/api/index.json` records no refusal for the folder.

The guard is the two I044 blocks in `tests/ir/test_fix_bounds.py`, each checked against the wrong
states rather than against one. For the source: the original condition fails the admitted case, the
reported origin of a genuine overrun (`offsets=(24, 0)`, which must still be refused, and at 1536
rather than the row-major 6144) and the end-to-end `pipesim` run; deleting the bounds check fails
the overrun and the UB control; extending NZ to L1 and UB alike fails the UB control. For the
destination: not naming the layout fails four of its five, and the fifth is the offset-zero control
that has to stay green in every state. Its helper builds the `Memory` with **no** layout on purpose,
because that is what the model hands `fix_errors` — a test that spelled `nz` there would have been
green against the row-major origin the defect was.

Qualified on an A5 card, both directions
(receipt). `examples/api/cube_quant_ub` passes
7/7 under `--launcher board` with cce, `split_i8` bitwise over all 2048 of its output bytes — the
row-offset L0C source the model had been skipping. The row-offset L1 destination is
`tools/diag/probes/l0c_to_l1_row_offset.py`, which inserts an FP32 accumulator at row 16 of a
`[32, 32]` FP16 Mat tile and reads it back through a second product against the identity: bit-exact
against the interpreter over all 1024 elements, and exact against the probe's own Python
expectation. A genuinely overrunning transfer was **not** run on the card — that is an
out-of-bounds on-chip write on a shared box, and the refusals are what the regression assertions
pin by number instead.

## Closed on 2026-09-23

A split-K int4 matmul sliced L1 in logical int4 elements instead of int32 carriers. `k` is
logical, but an int4 operand's L1 tile, its L0 slot and the `mem.reinterpret` that presents the
slot as int4 are all counted in carriers of 8 — the no-split path passes `A.span[1]`, the carrier
span, for exactly that reason. The split-K path passed `valid_k`, so every consumer was eight
times too wide and the L1 window began eight times too far along. Only `cube.mmad` stayed right,
because it takes the logical `K` directly, which is why the emitted sequence looked plausible next
to the shipped one.

Measured on an A2 card, no bias anywhere:

| case | before | after |
| --- | --- | --- |
| int4 K=128, no split (control) | exact | exact |
| int4 K=128, `splitk=64` (8 carriers) | **4089 of 4096 wrong** | exact |
| int4 K=256, `splitk=128` (16 carriers) | **4092 of 4096 wrong** | exact |
| fp16 K=128, `splitk=64` (control) | exact | exact |

Both chunk widths failed, so it was never about the width; the fp16 control is what placed it in
the int4 path rather than in split-K. **Nothing on a card had ever run this combination**: no
kernel in either repository passes `splitk=` with int4 operands — the only such call sites are the
canonical backend cases, which are compiled and never executed, and the pass tests. A form that
only ever compiles has no board result, the same way a form that never compiles does not.

It surfaced while qualifying the int4 fused bias under split-K, where the first run was wrong *and
so was its no-bias control* — the second time in one day that a failing control was the finding
rather than an obstacle. The guard is `tests/passes/test_a2_int4_splitk_carriers.py` (5 of 9
assertions fail without the fix; the 4 that pass are the mmad's logical `K`, the no-split control
and the fp16 control). The gallery emits byte-identically across the change: 109 of 114 kernels
compared, 0 moved.

[A2-B32-ND2NZ-L1-OVERWRITE](A2-B32-ND2NZ-L1-OVERWRITE.md) — an int32 L1 tile took the c220 cube's
fp32 ZZ layout. The cube keeps fp32 L1 tiles as ZZ and everything else as NZ; the backend selected
ZZ with `sizeof(T) == 4`, so an int32 tile — which is what an int4 operand's carriers are — was
written, read and address-folded as ZZ while `addr_alloc` and the model had it as NZ. ZZ needs
`align16(rows) * align16(cols) * width`, so a `[64, 8]` int32 tile wrote 4096 bytes into a 2048-byte
reservation and destroyed whichever neighbour came next. The old framework tests
`dst.dtype is Datatype.float`; the port kept its four L0A/L0B branches and lost the predicate.

Two diagnoses were wrong on the way, and both survived because they fit the measurements. "The
trigger is the 32-bit width" was refuted by an f16 tile that was also damaged. "`nd2nz` pads its
column count to 16 elements, dtype-independently" fit every number for compensating reasons — f16's
NZ granule really is 16 elements, and int32's 4096 came from ZZ rather than padding — and the
repair built on it (reserve `align16(cols) * rows * width`) is the ZZ footprint applied to every
tile, so the card went clean and the story held. Reading the CANN parameters it claimed to rest on
is what broke it: they say a `[64, 8]` int32 NZ tile writes 2048, leaving the measured damage with
no mechanism. **The arithmetic had said so all along.**

Fixed at the three sites that tested a width where they meant a dtype — `gm_to_l1_nd2nz` and
`l1_to_l0` in `tensorutils_cce.h`, and `elem_bytes_offset` in `backends/cce/views.py` — plus the
NZ b32 write the c220 branch had never needed. `addr_alloc` is re-derived from the layouts:
`align16(rows) * align_up(cols, G) * width`, `G = 16` for ZZ and `C0 = 32 / width` otherwise. That
is smaller than the old reservation for 32-bit integers and larger for two shapes it had been
under-reserving: int8 tiles whose column count is not a multiple of 32, and any tile under 16 rows.

Qualified on an A2 card **at the reduced reservation**, which is what separates a repair from a
mask: int4 carriers at 8 and 16 clean, fp32 at 8 / 16 / 64 columns clean, the int4 fused bias and
its split-K form exact. Six a5 kernels move (`online_mx` ×2, `simt_transpose` ×2, `a5_decode_fp8`,
`a5_mla`) and were re-run on an A5 card; the other 103 emit byte-identically. The guard is
`tests/passes/test_l1_tile_reserves_its_physical_footprint.py`, whose load-bearing assertion is
that an fp32 and an int32 tile of the same shape on the same device reserve **differently** — a
test keyed on the element width cannot say that, which is why the first one missed it.

A SIMT body could not call the shared floor remainder. `cce/host.py` prints
`ascrip::FloorMod<T>(a, b)` for an integer floor `%`, and
`backends/shared/include/scalar_math.h` declared that helper `__aicore__` alone, so bisheng
answered `candidate function not viable: simt_vf function can only call simt_callee function`
at the call site and the custom-op build failed before any kernel ran. Any SIMT kernel with a
floor remainder in it was therefore emittable and unbuildable; there is no board result from
before this date for one, because there could not be. The first attempt gave the one definition both
attributes and was wrong in the other direction -- bisheng answers `simt_callee function can
only be called by simt_vf/simt_callee function`, so an ordinary body could no longer call it,
which `kda_bwd`'s `scan_fused` proved on the same card. The attributes do not compose. The
helper is therefore defined twice, `ascrip::FloorMod` and `ascrip::simt::FloorMod`, and
`cce/host.py` picks the namespace from `self.fn.kind`, the way it already picks `__sqrtf`.

Qualified on an A5 card, both directions and both backends, against the corrected helper:
`kernels/ascriptor_kernels/algorithms/simt_transpose`, which calls it only from a SIMT body,
passes 14/14 under `--launcher board` with cce and again with pto_isa; `projects/a5/kda_bwd`,
whose `scan_fused` calls it only from an ordinary body, passes 5/5 under each. Neither built
under either backend before.

The guard is `tests/backends/test_simt_scalar_helpers.py`. One kernel writes a floor remainder
on both sides of the boundary, each call is checked against the declaration it resolves to, and
the two bodies are compared so they cannot drift. It fails on both wrong states: the original
`__aicore__`-only helper and the both-attributes one. The first attempt passed a four-demo
regression build that happened to contain no ordinary-body floor remainder, which is why the
guard checks the emitted text rather than a sample of demos.

It was found by running the demo gallery on hardware and never entered this register as open.

## Closed on 2026-09-22

M10-100 and D-260 met their closure conditions in this batch and have left this
directory. Their current rules moved into
[RFC-0013](../rfc/0013-pypto-native-synchronization.md): the static local capacity
bound is one-sided, and delegated credits are per Tile group rather than per byte
range. Their regression guards stay where they run -
`tests/backends/test_pypto_one_sided_bounds.py` and
`tests/backends/test_pypto_alias_group_credits.py` - and their hardware
qualification was the kernel owner's, in
historical kernel qualification records; both were removed when that repository
became a demo gallery on 2026-09-23 and resolve at kernels
`f79b44b721eba5080b802409c2388f74638405ce`, as dated provenance with no successor. The records themselves are
recoverable as `defects-closed-20260922`; see
closed defects. IDs are not reassigned.

## Lifecycle and recovery

Keep only unresolved implementation/model defects here, with their reproducer,
affected scope, mitigation and closure condition. A scoped repair must leave any
remaining in-scope failure explicit. Accepted unsupported forms belong to their
API/backend contract; dependency defects belong to the upstream report.

After closure, retain the regression and necessary qualification evidence, move
current rules to their owning API/RFC, and update consumers before removing the
investigation. Preserve exact historical bytes through a verified Git snapshot;
do not reuse defect IDs. The 138 closed or retired records removed during this
cleanup are available through closed-defect recovery.
Historical executions retain their original failures and qualification scopes.

## All-vector barrier emission against the current PyPTO API

A full grouped integer matmul/SwiGLU workload exported the adjacent all-vector
ready/wait pair as `sync_all(core_type=AIV_ONLY, mode=SyncAllMode.HARD)`.
The installed PyPTO package exposes neither the `SyncAllMode` enum nor a `mode`
parameter. Its parser consequently rejected the enum expression before vendor
compilation. The current documented API is `sync_all(*, core_type=MIX)` and
uses FFTS hardware synchronization. The emitter now passes only
`core_type=AIV_ONLY`; the IR participant set and ordering are unchanged.
The plugin regression compiles real vector and mixed DSL kernels, checks the
generated calls against that keyword-only signature, and rejects an unpaired
wait. Numerical and device qualification belongs to the task's retained
full-workload runs; source emission alone is not hardware acceptance.
