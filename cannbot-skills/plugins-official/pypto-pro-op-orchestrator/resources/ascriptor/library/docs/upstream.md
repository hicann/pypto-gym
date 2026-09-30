# Unified upstream issue report

Maintained by Ascriptor library. Updated **2026-09-23**. This is the single
issue document for reporting to the PyPTO Pro, PTO ISA and relevant CANN
development teams. Coverage/mapping pages and case contracts reference its
IDs; they do not maintain competing issue lists. No upstream submission has
been made by this documentation update.

## Scope and evidence

The September 7 dependency checkpoint matches PyPTO Python revision
`86ef830cb68110cf4092420a4d20c2a664677530` byte for byte across 65 files.
[Dependency fingerprints][dependencies] also identify the loaded compiler
libraries, bisheng binary and eight PTO headers. The package's 0.2.1 marker
alone does not identify those builds. These findings concern that inspected
checkpoint, not a claim about the latest upstream HEAD.

The prototype report was written September 3 and revised through September 5,
against CANN 9.2.0 inner V100R001C25B046, PyPTO 86ef830 and Ascend950PR.
Its original content was byte-identical in the prototype, backup and imported
successor document. [Reconciliation provenance][reconciliation] records its
SHA-256, every original row's disposition, and the current source inspection.
Historical measurements retain their original date/device/corpus; the September 7
consolidation performed no new performance or hardware run. A5-UP-036 references
a separate September 21 native acquisition against PyPTO source build `fe20d72`,
with its own runtime and artifact identities. Its PyPTO 0.2.1 marker does not
establish equality with the earlier 65-file fingerprint or a later upstream HEAD.

**The current installation is PyPTO 0.3.0**, built on the A5 box on 2026-09-23 from official
`acabcaa780cc4cfdec7df91b2e9183d99bfb7fc2` (`gitcode.com/cann/pypto`, branch `9.2.0`) with
`docs/patches/pypto-pro-integer-cast.patch` applied and declared through
`ASCRIPTOR_PYPTO_SUPPLEMENTS=integer-cast`. The two revisions above are 448 commits behind it
and are retained because the findings dated to them were measured there; they are not what a
board run uses now.

**A SIMT emission cannot run on either of them.** `backends/pypto_pro/emit.py` opens a SIMT body
with `@pl.vector_function(mode="simt", max_threads=N)`, and that signature arrives upstream in
`2b49dbfa` on 2026-09-15 — after `86ef830c` (09-02) and `fe20d72` (08-31), both of which take the
bare `vector_function(fn)` and answer `TypeError: unexpected keyword argument 'mode'` on every
case. So 0.3.0 or later is the floor for the SIMT path, whatever an issue below was measured
against. Two findings were re-checked on it: A5-UP-006 still refuses the scaled FP32-to-BF16
fixpipe, and a 64x48 SIMT transpose loop failure no longer reproduces and has been removed.

Evidence labels mean:

- **Current:** dated successor API/source, native compile or board evidence
  as specified in the issue; September 7 unless a later receipt is identified.
  Source inspection is not a board pass.
- **Source:** a specific upstream API/validator/header restriction inspected
  in the fingerprinted files; its requested extension has not been board-qualified.
- **Historical:** prototype silicon/compiler/performance evidence. A suffix
  **source** confirms a mechanism in the current source; **guard** confirms
  only that Ascriptor still protects the old failing form. Neither is a fresh
  native reproduction.
- **Workaround:** the named domain has an Ascriptor implementation. The native
  upstream limitation may still be worth fixing, but the supported domain is
  not counted as an unavailable kernel.

The September 7 [canonical matrix][matrix] is **306 cases x 3 backends = 918 keys**.
Its 43 PyPTO upstream-gap keys are A5-UP-004: 17, A5-UP-005: 16,
A5-UP-006: 5, A5-UP-008: 3, and 2 for a SIMT transpose restriction that upstream has
since fixed and that no longer has an entry here. CCE and PTO have zero upstream-gap
keys there. The three M10-059 numerical failures remain failed, once per backend.
Other issue IDs below do not add missing cases to this matrix. The prototype's
**43 refused kernels out of 142** is a different historical population; the
equal number 43 is coincidental, and the two counts must not be added or equated.

The historical counts on this page describe an earlier corpus and are not current
qualification evidence. This first source snapshot contains no kernel-side device
receipts. To establish a present result, rerun the relevant demo folder on the local
target machine and keep its evidence with the source ID.

## Issue index

The default [native synchronization mode](rfc/0013-pypto-native-synchronization.md)
has a separate capability boundary: `auto_mutex=True` manages local Tile locks;
it does not implicitly run the cross-core `pipeline` stage transformation.
A September 10 mixed-core GM publication probe retained identical computation,
14 `get_buf`/`rls_buf` pairs and two local barriers in both variants. Keeping the
nine generated cross-core set/wait calls passed 16 fresh/reused-buffer trials;
removing only those calls failed all 16, with all 1,536 outputs reading the
poisoned old workspace generation. Final workspace publication, inputs and guards
were correct. Actual build C++ matched the separately reviewed generated source.
The no-stage `pipeline=PipelineConfig(sync_only=True)` variant was explicitly
rejected by the installed analyzer. This establishes a missing automatic
publication protocol in that configuration, not a defect in a promised cross-core
`auto_mutex` capability or a new canonical matrix gap. Cross-core takeover needs
an explicitly proved stage transformation; local barriers alone are insufficient.

A5-UP-001 through A5-UP-008 keep their existing IDs. Additional inherited,
source-inspected and native findings receive IDs below. A zero in the final column means
no gap key in the canonical matrix, not universal support or a resolved issue.
The index contains **45 issue IDs**; A5-UP-036 through A5-UP-046 add no gap key to that historical matrix.

| ID | Issue | Evidence | Canonical gap keys |
| --- | --- | --- | ---: |
| [A5-UP-001](#a5-up-001) | Ordinary scalar abs | Current | 0 |
| [A5-UP-002](#a5-up-002) | Ordinary scalar sqrt | Current | 0 |
| [A5-UP-003](#a5-up-003) | Dynamic scalar cast and loop induction dtype, refused inside a vector function | Current + historical performance | 0 |
| [A5-UP-004](#a5-up-004) | NZ Mat padding fill cannot reach TFILLPAD | Current | 17 |
| [A5-UP-005](#a5-up-005) | Native img2col / convolution L1-to-L0A load | Current | 16 |
| [A5-UP-006](#a5-up-006) | Scaled FP32-to-BF16 FIX store | Current | 5 |
| [A5-UP-007](#a5-up-007) | Partial-row FP32 L0C-to-L1 transfer | Current | 0 |
| [A5-UP-008](#a5-up-008) | Scalar-scaled FP32-to-FP32 / E4M3 Acc-to-Vec | Current | 3 |
| [A5-UP-010](#a5-up-010) | L1 TExpandsTile repeat/block split | Historical performance | 0 |
| [A5-UP-011](#a5-up-011) | Redundant rank-2 MTE3 loop configuration | Historical performance + source | 0 |
| [A5-UP-012](#a5-up-012) | PyPTO JIT versus per-core CANN build code generation | Historical performance | 0 |
| [A5-UP-013](#a5-up-013) | Split b8 transposing L1-to-L0B extraction | Historical performance + source | 0 |
| [A5-UP-014](#a5-up-014) | Native i64/u64 VF arithmetic register forms | Historical + guard | 0 |
| [A5-UP-015](#a5-up-015) | Complex dtype and immediate surface | Historical + source | 0 |
| [A5-UP-016](#a5-up-016) | Scaled INT32-to-BF16 FIX store | Historical + source | 0 |
| [A5-UP-017](#a5-up-017) | Unsigned 8-bit FIX requantization | Historical + source | 0 |
| [A5-UP-018](#a5-up-018) | Hybrid HiFloat8 FIX-store rounding selector | Historical + source | 0 |
| [A5-UP-019](#a5-up-019) | DN destination for L0C-to-GM FIX store | Historical + source | 0 |
| [A5-UP-020](#a5-up-020) | FIX scale silently lost with a source offset | Historical + partial workaround | 0 |
| [A5-UP-021](#a5-up-021) | FP16 / HiFloat8 VF hybrid or odd rounding | Historical + guard | 0 |
| [A5-UP-022](#a5-up-022) | BF16-to-FP16 VF conversion argument mismatch | Historical + guard | 0 |
| [A5-UP-023](#a5-up-023) | 64-bit VF conversion lane layout and overloads | Historical + guard | 0 |
| [A5-UP-024](#a5-up-024) | Widening VF casts from packed i4 | Historical + guard | 0 |
| [A5-UP-025](#a5-up-025) | SIMT index typing in atomic exchange/CAS and bit counts | Historical + guard | 0 |
| [A5-UP-026](#a5-up-026) | b8 gather widening semantics | Historical + guard | 0 |
| [A5-UP-027](#a5-up-027) | Unaligned UB cursor at a runtime origin | Historical + guard | 0 |
| [A5-UP-028](#a5-up-028) | Arbitrary DMA padding value | Historical + source | 0 |
| [A5-UP-029](#a5-up-029) | Non-unit innermost GM-to-UB stride | Historical + guard | 0 |
| [A5-UP-030](#a5-up-030) | Native higher-rank DMA descriptor preservation | Historical + workaround | 0 |
| [A5-UP-031](#a5-up-031) | Float immediates printed with six decimal places | Historical + workaround | 0 |
| [A5-UP-032](#a5-up-032) | Native VF register groups | Source | 0 |
| [A5-UP-033](#a5-up-033) | Native integer VF floor remainder | Source | 0 |
| [A5-UP-034](#a5-up-034) | Scalar access to storage-only low-precision dtypes | Source | 0 |
| [A5-UP-035](#a5-up-035) | PTO Mat TFILLPAD layout/value restrictions | Source | 0 |
| [A5-UP-036](#a5-up-036) | Loop-carried integer snapshots lost in native C++ | Current native + workaround | 0 |
| [A5-UP-037](#a5-up-037) | INT32 scatter casts UINT32 offsets to signed registers | Current native compile + composite workaround | 0 |
| [A5-UP-038](#a5-up-038) | Native mutex pipe selection for Cube scalar fill | Fixed upstream `6a652e733` | 0 |
| [A5-UP-039](#a5-up-039) | Acc-to-Mat insert mutex uses MTE3 instead of FIX | Native failure + scoped kernel workaround | 0 |
| [A5-UP-040](#a5-up-040) | Documented `mrgsort2` argument order differs from its graph operands | Current native + source | 0 |
| [A5-UP-041](#a5-up-041) | Attribute-free ops skip keyword validation the CCE emitter relies on | Source | 0 |
| [A5-UP-042](#a5-up-042) | Generated SIMT function names collide with libm overloads | Current native compile | 0 |
| [A5-UP-043](#a5-up-043) | FP16 `exp_sub` predicate documentation contradicts A5 | Current native | 0 |
| [A5-UP-044](#a5-up-044) | `vf.load` / `vf.store` ignore their declared keywords | Source | 0 |
| [A5-UP-045](#a5-up-045) | `syncthreads` under runtime control flow compiles despite its documented restriction | Current native compile | 0 |
| [A5-UP-046](#a5-up-046) | `vf.full` of an FP4 register compiles to a `vbr` call with no FP4 overload | Current native compile | 0 |

## Issue details

<a id="a5-up-001"></a>

### A5-UP-001 — Ordinary scalar abs

Owner: **PyPTO Pro**. Evidence: **Current**. Origin: new successor finding.

`scalar_abs(Var)` / `scalar.abs` has no ordinary kernel/VF scalar entry point.
`ir/op/scalar_ops.py` registers min/max/const; `pl.abs` takes Tiles and
`pl.simt.abs` is confined to SIMT functions. **This is distinct from register
`vf.abs` / `Vf.abs`, which has an existing mapping.** `mode="vec"` chooses AIV
execution; it does not turn a scalar operation into a VF register operation.
The six i8/i16/i32/i64/f32/FP16-widening [scalar abs cases](../examples/api/scalar_abs)
pass CCE/PTO on A5; PyPTO refuses at the located scalar opcode.
Ordinary BF16 scalar abs is now also supported by the CCE/PTO library lowering
through explicit scalar conversions; PyPTO's missing ordinary scalar operation
is unchanged (repair receipt (`docs/migration/fragments/defect-repairs-20260907.json`)).
**Request:** expose a typed, native ordinary scalar absolute-value operation.
Evidence: M10-062, [scalar receipts][scalar].

<a id="a5-up-002"></a>

### A5-UP-002 — Ordinary scalar sqrt

Owner: **PyPTO Pro**. Evidence: **Current**. Origin: new successor finding.

`pl.sqrt` is a Tile operation and the scalar SIMT handler rejects ordinary
kernel/VF scope. This says nothing about VF register sqrt support. FP32 and
FP16 [scalar sqrt cases](../examples/api/scalar_math) pass native
CCE/PTO board checks; the PyPTO ordinary scalar form is absent.
CCE/PTO ordinary BF16 sqrt and conversion are now qualified by the library's
explicit conversion lowering; this does not supply the missing PyPTO API.
**Request:** expose native ordinary scalar sqrt with explicit dtype semantics.
Evidence: `ir/op/scalar_ops.py`, `ir/op/simt_ops.py`, [scalar receipts][scalar].

<a id="a5-up-003"></a>

### A5-UP-003 — Dynamic scalar cast and loop induction dtype

Owner: **PyPTO Pro**. Evidence: **Current + historical performance**. Origin: P3, D-139, D-150.

The original inspected package exposes literal-only `pl.const`, Tile-only
`pl.cast`, and SIMT-only `pl.simt.cast`. The local integer
[compatibility supplement](pypto-pro-supplements.md#1-integer-scalar-cast)
adds `pl.cast(value, dtype)` to ordinary kernel/VF code using the existing IR
Cast. It is recorded by exact dependency hashes in the
cast receipt, not claimed for every official
package. Float conversion and automatic preservation of loop-index width remain
separate gaps. Ascriptor now restores narrow VF div/mod operands and maps explicit
integer casts through M10-090.
This is separate from same-dtype UB getval/setval, VF register casts and A5-UP-025.
The prototype also measured runtime `pl.range` induction variables becoming
`uint64_t` despite narrowed bounds or VF parameter annotations. Manually
narrowing 15 scalar min/max operands in generated C++ changed `fd_modified`
from 13.75 to 8.96 us. Typed literals, typed bounds and parameter annotations
did not remove that historical cost. Ascriptor's D-155/D-156 rewrites removed
that kernel's bottleneck; this is **not a current performance result**.
The current VF code generator narrows the final C++ loop variable to uint16_t,
but its IR INDEX type can still promote arithmetic on other operands to 64 bits.

Upstream `289942aa3` (2026-09-21) has neither half: `_ir_cast` is unchanged, and
`e70d321dd` (2026-09-10) added `_VF_SCALAR_PL_OPS`, a whitelist that admits only
`pl.range/min/max/const` inside `@pl.vector_function` and refuses every other
`pl.*` — including upstream's own scalar `pl.astype`. So a vf body has no scalar
integer conversion at all on a stock installation, and the supplement now carries
both halves: the Scalar+DataType dispatch and `cast` in that whitelist. The
2026-09-22 port applies cleanly to `289942aa3` and the board probe checks both.
**Request:** upstream the ordinary scalar cast, admit a scalar conversion inside a
vector function, and preserve/declare loop-index width. Evidence: the scoped cast
receipt and prototype P3/D-139/D-150.

<a id="a5-up-004"></a>

### A5-UP-004 — NZ Mat padding fill cannot reach TFILLPAD

Owner: **PyPTO Pro / PTO**. Evidence: **Current**. Origin: P2, D-153.

`pl.fillpad` rejects a null-pad destination. A zero-pad destination produces
a separate null-pad source alias; the Mat `TFILLPAD` overload requires one
shared Tile type, while the two-type overload requires Vec tiles. Direct
native PyPTO probes of normal/in-place modes fail compilation without
Ascriptor. The prototype additionally tried all three FillPadMode values.
This blocked 12 matrix-block-quant cases and five block-absmax cases. It is
related to A5-UP-010's L1 fill performance, but the two defects are distinct.
**Request:** retain source pad type or provide a compatible two-type Mat
overload. PTO ISA positive-zero Mat fill works; see A5-UP-035 for its limits.
Evidence: `npu/a5/TFillPad.hpp`, `pto_instr.hpp`, [matrix][matrix], [M10-063][adapters].

**2026-09-22 — the two canonical users stopped asking for a padding-only fill.**
The defect itself is unchanged and unfixed: `dma.fillpad_l1` still has no `pl`
spelling, and any kernel that needs to clear only an NZ tile's padding still has
no native route. What changed is on our side. `matrix_normalization/block_absmax`
and `matrix_block_quant` now clear the whole L1 A-slot with
`dma.set_constant_to_l1`, which `pl` does reach (`pl.expands` on a Mat tile,
behaviour #31 of the [mapping](pypto-pro-mapping.md)), paying A5-UP-010 /
D-152's write amplification — 12,910,592 fill bytes across the 17 cases where the
padding-only form wrote 540,672, which the pipe model prices at between +0.76% and
+4.33% of each case's cycles — in exchange for support. Eight of those 17
cases now EMIT for pypto_pro (`block_absmax_minimum`, `block_absmax_source`,
`default_source`, `default_reuse`, `default_midpoints` and the three `pack4`
peers); the other nine refused on an unrelated Ascriptor limitation — a
runtime-sized `mem.slice` window had no tile form — which applies when M is not a
multiple of the M tile. An earlier study reported matching A5 and CCE results
for those eight; this snapshot contains no device receipt for that result. Later
the same day library `5a3552d` repaired the other nine as well — a vector
function's parameter is a bare pointer, so the base tile spells that window and
`pl.make_tile` was never on the path. All 17 now have an emission route; rerun
hardware checks before asserting current device behavior. The 17 canonical gap keys in the September 7 matrix stay
as recorded: that matrix is historical evidence of the sources it ran, not a live
count.

**2026-09-22 — `dma.fillpad_l1` is retired; this entry stays open.** With both
canonical users on the whole-tile fill the op had no caller, one backend that
could print it and a permanent refusal on the other two, so it was deleted from
the IR, the surface and every backend. Nothing upstream changed: a kernel that
needs to clear only an NZ tile's padding still has no `pl` route, and this entry
is the record of that. It is what a future padding-only fill would be reopened
against — restoring the op is cheap next to the upstream fix it waits for, and
`ir/ops/dma.py` at library `f75ee23^` carries its exact definition.

<a id="a5-up-005"></a>

### A5-UP-005 — Native img2col / convolution L1-to-L0A load

Owner: **PyPTO Pro**. Evidence: **Current**. Origin: R4.

The inspected Python language and op-registration surface exposes no native
equivalent for `dma.l1_to_l0.img2col` / `cube.conv2d`. All 16 cases in the
canonical native convolution unit identify the missing source operation.
This is an API reachability finding, not a claim that A5 lacks convolution
instructions or that every underlying PTO layer has been exhaustively searched.
**Request:** expose the native load/descriptor form. Evidence: [matrix][matrix]
and the adapter's `_PROVEN_ABSENT` dispatch.

<a id="a5-up-006"></a>

### A5-UP-006 — Scaled FP32-to-BF16 FIX store

Owner: **PyPTO Pro**. Evidence: **Current**. Origin: R3: FP32-to-BF16.

`ir/op/block_ops.py::_check_scale_dst_supported` explicitly rejects scaled
FP32-to-BF16 output. Five canonical KDA backward cases need this form;
their CCE/PTO executions pass. The upstream error's hardware explanation is
not adopted as a blanket hardware limitation. A5-UP-016 covers the separate
INT32 source form checked by the same validator.
**Request:** expose the supported FIX conversion with its scale preserved.
Evidence: current validator source and [matrix][matrix].

<a id="a5-up-007"></a>

### A5-UP-007 — Partial-row FP32 L0C-to-L1 transfer

Owner: **PTO ISA**. Evidence: **Current**. Origin: new successor finding.

TINSERT/TMOV select channel splitting but copy the declared source rows;
TEXTRACT retains valid rows but disables splitting. A partial-row FP32
transfer therefore has no equivalent among the inspected forms.
Whole-row FP32 transfers are supported through TINSERT and are excluded.
**Request:** support channel splitting together with runtime valid rows.
Evidence: `npu/a5/TInsert.hpp`, `TMov.hpp`, `TExtract.hpp`, [M10-063][adapters]
and [dependency fingerprints][dependencies]. No canonical case is counted here.

<a id="a5-up-008"></a>

### A5-UP-008 — Scalar-scaled FP32-to-FP32 / E4M3 Acc-to-Vec

Owner: **PyPTO Pro / PTO**. Evidence: **Current**. Origin: new successor finding.

Generated C++ calls the scalar TMOV overload. PTO's dtype selector chooses
NoQuant for FP32 (the board control returns the unscaled product) or a
vector-quantization mode for E4M3. The Python move API has no explicit mode
parameter. PTO ISA now calls the existing helper with the correct explicit
mode and passes both controls; three canonical V1 cases remain PyPTO gaps.
**Request:** select the scalar quantization mode correctly or expose it in
Python. This is separate from offset/scale loss in A5-UP-020.
Evidence: [M10-063][adapters], [scalar/control receipts][scalar], [matrix][matrix].

<a id="a5-up-010"></a>

### A5-UP-010 — L1 TExpandsTile repeat/block split

Owner: **PTO, reached through PyPTO Pro**. Evidence: **Historical performance**. Origin: P1, D-152, D-159.

Prototype `npu/a5/TExpandS.hpp::TExpandsTile` used
`repeatConfig = (1 << 16) | repeatTimes`; CCE used
`(n_blocks << 16) | 1` for the same L1 fill. A 16 KB fill incurred 512
transactions rather than one. Header-only block/repeat correction preserved
output bits: `v8_allhif8` 20.72 -> 17.73 us, L1 writes 2098.7 -> 909.2 KB
(CCE 905.9 KB); `matmul_chunk_absmax_norm128` 179.6 -> 93.2 us.
P1/P2 jointly accounted for 112.3 of 140.6 us in that historical measurement;
this must not be projected onto the successor matrix or a different sweep.
**Request:** encode the legal block count, as TFillPad/MGather do, instead
of one block per repeat. A5-UP-004 prevents the narrower padding-only route.
The old silicon result is retained; TExpandS and current timing were not remeasured here.

<a id="a5-up-011"></a>

### A5-UP-011 — Redundant rank-2 MTE3 loop configuration

Owner: **PTO**. Evidence: **Historical performance + source**. Origin: P4, D-141.

`TStoreVecND` / `TStoreVecDN` unconditionally program loop1/loop2 stride,
loop size and normal-mode restoration even when both outer dimensions are 1.
Those writes are still visible in the fingerprinted `npu/a5/TStore.hpp`.
The prototype's four-call removal changed `fd_modified` MTE3 cycles
14273 -> 13110 and MTE3 ratio 0.181 -> 0.164, with no measured wall-time gain
because MTE3 was not its critical path. **Correctness is not blocked.**
**Request:** omit redundant rank-2 configuration when state requirements
permit it. Current latency has not been remeasured.

<a id="a5-up-012"></a>

### A5-UP-012 — PyPTO JIT versus per-core CANN build code generation

Owner: **PyPTO Pro build pipeline / CANN**. Evidence: **Historical performance**. Origin: P5, D-143, D-144.

Prototype same-card timings were CCE/CANN op build 8.513 us, PTO ISA/CANN
op build 8.674 us and PyPTO/direct bisheng 8.956 us. The same CCE source
also slowed under direct compilation, with the compiler binary checked equal.
The observed build difference was per-core dav-c310-vec/cube compilation
with auto-sync on versus one dav-c310 fatbin with auto-sync off. Optimization
levels, drivers, output forms, arch variants, tile-fusion flags and nine LLVM
switches were tried; the exact causal option was not isolated.
**Request:** align the supported JIT build path or identify the missing
compiler configuration. Keep this as a measured build discrepancy, not a
proven claim that auto-sync alone causes the loss. No successor perf rerun.

<a id="a5-up-013"></a>

### A5-UP-013 — Split b8 transposing L1-to-L0B extraction

Owner: **PTO**. Evidence: **Historical performance + source**. Origin: P6, D-165, D-166.

`TExtractToBTransCompact` takes the b8 branch solely from dtype/compact mode
and splits mStep=16 into eight mStep=2 loads. That branch and loop remain in
the fingerprinted `npu/a5/TExtract.hpp`. For prototype `mla_hif8`'s
(256,512) Mat -> (256,128) Right move, CCE issued one legal transposing load.
A helper differing only by loop removal preserved output bytes, reduced
MTE1 cycles per move 152.2 -> 131.4 and kernel time about 312.8 -> 306.8 us.
The reversed-shape ZN alias did not fit this matmul operand's required shape.
**Request:** split only when the hardware geometry requires it. Current
source confirms the mechanism; the measured improvement remains historical.

<a id="a5-up-014"></a>

### A5-UP-014 — Native i64/u64 VF arithmetic register forms

Owner: **PyPTO Pro / PTO VF interface**. Evidence: **Historical + guard**. Origin: R1.

The prototype's largest reachability group was 16 kernels using i64/u64
register add/reduce/compare/dup/shift/gather forms. Current
`VfPrinter.check_dtype64` retains the located refusal. DT_INT64/DT_UINT64
scalar/storage declarations do not provide the native register-pair carrier
used by CCE compiler overloads. This is distinct from ordinary scalar abs
and from register casts (A5-UP-023).
**Request:** expose the native carrier/overloads; do not silently reduce the
lane count or substitute unqualified low/high arithmetic emulation.
No fresh successor board sweep of those 16 prototype kernels is claimed.

<a id="a5-up-015"></a>

### A5-UP-015 — Complex dtype and immediate surface

Owner: **PyPTO Pro**. Evidence: **Historical + source**. Origin: R2.

The inspected DT_* set has no c32/c64/complex immediate form; `_pl_dt`
retains the explicit diagnostic. Eight prototype kernels were affected.
Integer/float carrier storage alone does not establish complex arithmetic
semantics. **Request:** define the complex dtype/register/constant surface
or a supported equivalent. This is outside the canonical matrix.

<a id="a5-up-016"></a>

### A5-UP-016 — Scaled INT32-to-BF16 FIX store

Owner: **PyPTO Pro**. Evidence: **Historical + source**. Origin: R3: INT32-to-BF16.

The current `_check_scale_dst_supported` validator rejects INT32-to-BF16
when quantization is active, alongside the FP32 form in A5-UP-006. The
prototype retained a failing form in its six-kernel FIX group; there is no
fresh successor board result for this source dtype.
**Request:** provide the scaled INT32-to-BF16 path with defined rounding,
or document its exact backend restriction. Do not count it as another five
KDA cases: those use FP32 and already belong to A5-UP-006.

<a id="a5-up-017"></a>

### A5-UP-017 — Unsigned 8-bit FIX requantization

Owner: **PyPTO Pro**. Evidence: **Historical + source**. Origin: R3: u8 requant.

The current `_check_scale_dst_supported` validator rejects UINT8 quantized
output. Prototype CCE ran the corresponding u8 requant case; the Python
diagnostic's assertion that hardware has no unsigned path is therefore
not sufficient to classify hardware support.
**Request:** expose the supported form and preserve scale/offset packing,
or identify the precise upstream limitation. Current source confirms the
rejection; the CCE/PyPTO paired board evidence remains historical.

<a id="a5-up-018"></a>

### A5-UP-018 — Hybrid HiFloat8 FIX-store rounding selector

Owner: **PyPTO Pro**. Evidence: **Historical + source**. Origin: R3: hif8_hybrid.

`pl.store` has no equivalent of the `hif8_hybrid` FIX rounding attribute;
the adapter's `_ATTR_ABSENT_UPSTREAM` guard retains it. This is **not** the
VF f16/HiFloat8 cast-rounding issue in A5-UP-021 and is not M10-059's exp
midpoint numerical comparison.
**Request:** expose the FIX hybrid rounding mode with its scale semantics.
The existing hardware observation is from the prototype, not a new run.

<a id="a5-up-019"></a>

### A5-UP-019 — DN destination for L0C-to-GM FIX store

Owner: **PyPTO Pro**. Evidence: **Historical + source**. Origin: R3: nz2dn.

For `dma.l0c_to_gm.nz2dn`, the C++ store has a DN form, but the prototype
`pl.DN` annotation left bytes untransposed and `pl.store(order=[1,0])`
refused. The current Python validator still requires ascending order;
the adapter retains the located refusal.
**Request:** expose a store layout/order that reaches the DN destination.
This finding does not concern supported DN-to-NZ loads. No fresh board retest.

<a id="a5-up-020"></a>

### A5-UP-020 — FIX scale silently lost with a source offset

Owner: **PyPTO Pro / PTO code generation**. Evidence: **Historical + partial workaround**. Origin: R3: scale-plus-offset, D-128.

The prototype's native `pl.move(dst, src, scale=s, offset=[r,c])` lowered
to TEXTRACT without a scale operand; the offset-free spelling reached
scaled TMOV. It built but returned unscaled data for the offset sub-block.
The successor now represents **full-height NZ column strips**, including
rotating slots, as address aliases and uses offset-free scaled TMOV. Those
repaired shapes are no longer missing; [M10-063][adapters] and the current
matrix retain their evidence. Remaining scale-plus-window requests are
guarded; alternate representations have not been exhausted for every shape.
**Request:** preserve scale in the offset form or reject it upstream instead
of silently dropping it. A5-UP-008 is a different dtype-selector defect.

<a id="a5-up-021"></a>

### A5-UP-021 — FP16 / HiFloat8 VF hybrid or odd rounding

Owner: **PyPTO Pro / PTO VF conversion**. Evidence: **Historical + guard**. Origin: R5: f16/hif8.

Prototype `vf.astype` refused CAST_ODD/CAST_HYBRID for the FP16/HiFloat8
pair. The current `vf.cast` guard retains that evidence. Ordinary pair
conversions and FP32/HiFloat8 hybrid conversion must be assessed separately;
earlier blanket HiFloat8 refusals were repaired.
**Request:** implement the documented rounding forms for this dtype pair.
No new successor board reproduction of these rounding modes is claimed.

<a id="a5-up-022"></a>

### A5-UP-022 — BF16-to-FP16 VF conversion argument mismatch

Owner: **PTO VF code generation**. Evidence: **Historical + guard**. Origin: R5: bf16/f16, D-093.

Prototype generated `vcvt(dst,src,mask,ROUND_R,MODE_ZEROING)` omitted the
RS argument required by the BF16-to-FP16 form, causing a compiler
static_assert. The reverse FP16-to-BF16 direction worked. The current
adapter keeps the dtype-pair guard.
**Request:** emit the correct RS/round/mode argument sequence for this
overload. Evidence is historical compiler output, not a new compile.

<a id="a5-up-023"></a>

### A5-UP-023 — 64-bit VF conversion lane layout and overloads

Owner: **PyPTO Pro / PTO VF code generation**. Evidence: **Historical + guard**. Origin: C1, C2, D-221.

Prototype `cast_b64_widen` built and ran but i32->i64 returned source lanes
[0,2,4,6,...]; i64->i32 wrote alternate lanes with zero gaps. CCE was
bit-exact on the same inputs. These integer forms use the register-pair
`vcvt(dst,src)` with no mask/PART/MODE, unlike ordinary masked casts.
The distinct f32->i64 form needs `vcvt(dst,src,ROUND,RS)`; the masked PART
spelling instead caused an RS-position static_assert (C2).
The successor's broad 64-bit cast guard covers both, including the C2 pair;
the per-pair diagnostic branch is not an independent passing test.
**Request:** select each native pair overload and preserve lane k -> lane k.
This remains a historical silent-wrong-answer/compiler issue, not a new
board result or a claim that every possible 64-bit pair was measured.

<a id="a5-up-024"></a>

### A5-UP-024 — Widening VF casts from packed i4

Owner: **PyPTO Pro / PTO VF code generation**. Evidence: **Historical + guard**. Origin: C3.

Prototype `cast_i4_widen` selected generic vcvt with a uint8 register;
the packed `vector_s4x2` source requires dedicated `vcvt_s42f16`,
`vcvt_s42bf16` or `vcvt_s42s16` forms. Compilation found no matching
overload. Narrowing f16/i16 -> i4 worked and is excluded.
**Request:** expose the packed source type and correct widening intrinsics.
The current source-side guard preserves the historical failure evidence.

<a id="a5-up-025"></a>

### A5-UP-025 — SIMT index typing in atomic exchange/CAS and bit counts

Owner: **PyPTO Pro SIMT typing**. Evidence: **Historical + guard**. Origin: R6.

Four prototype rounds found value operands typed as index, with
`pl.simt.cast` refusing index conversion. The current emitter guards atomic
exch/cas and ffs/popc paths. Other atomic forms use contextual literals or
typed runtime values and are not part of this blanket claim.
**Request:** retain the intended operand dtype or support the necessary
index conversion. This is separate from ordinary scalar cast A5-UP-003.
No fresh native replay here.

<a id="a5-up-026"></a>

### A5-UP-026 — b8 gather widening semantics

Owner: **PyPTO Pro VF interface**. Evidence: **Historical + guard**. Origin: R7.

Prototype b8 gather widening exposed zero-extension and a board difference
on the signed case; the emitter currently rejects cross-dtype gather_copy.
That conservative guard also covers other cross-dtype pairs, which must
not all be described as independently measured failures.
**Request:** expose signed/unsigned widening explicitly and qualify each
pair. Same-dtype gather and 64-bit block gather are separate forms.

<a id="a5-up-027"></a>

### A5-UP-027 — Unaligned UB cursor at a runtime origin

Owner: **PyPTO Pro unaligned VF interface**. Evidence: **Historical + guard**. Origin: R8.

For `vf.ub_cursor`, a nonzero runtime starting position materialized a Var
that the prototype unaligned-pointer code generator rejected; it required
a bare tile. The current guard remains.
**Request:** accept a typed runtime origin for the unaligned cursor.
This does **not** prohibit ordinary strided `vf.load_align/store_align`:
those support `base + offset` pointer arithmetic (D-116), and the old
cursor-walk-only restriction on those operations is superseded.

<a id="a5-up-028"></a>

### A5-UP-028 — Arbitrary DMA padding value

Owner: **PyPTO Pro**. Evidence: **Historical + source**. Origin: R9.

`pl.TilePad` represents null/zero/max/min modes, not an arbitrary value;
the Python Tile declaration/load surface exposes no equivalent of the
requested padding value or C++ Tile's SetPadValue. The prototype declaration
probe was rejected and the current load-attribute guard retains it.
**Request:** expose the pad value separately from its mode, including
bit-preserving treatment where required. This is distinct from Mat fill
compilation (A5-UP-004) and PTO Mat fill's zero-only form (A5-UP-035).

<a id="a5-up-029"></a>

### A5-UP-029 — Non-unit innermost GM-to-UB stride

Owner: **PTO / PyPTO Pro DMA surface**. Evidence: **Historical + guard**. Origin: R10.

The prototype `loop_src_stride=[4,48]` gather-shaped load has no equivalent
in TLoadVecND2ND's contiguous innermost burst. The existing diagnostic
records `lenBurst=validCol*sizeof(T)` and a Vec ND row-alignment failure.
Outer-loop decomposition cannot supply that missing innermost element stride.
**Request:** expose the native NDDMA element-stride form or a semantically
equivalent supported DMA composition. No fresh successor replay of this
exact layout; do not confuse it with repaired rank/axis selection.

<a id="a5-up-030"></a>

### A5-UP-030 — Native higher-rank DMA descriptor preservation

Owner: **PyPTO Pro code generation**. Evidence: **Historical + workaround**. Origin: E3, D-131.

Prototype rank-3 shape [2,2,8] / strides [96,16,1] became TileShape2D,
dropping the outer 96 stride; a raw-pointer spelling behaved the same.
Ascriptor's `_nd_unroll` supplies supported cases as 2-D loads, requiring
contiguous innermost runs, full destination rows and aligned outer steps.
D-131's repaired `gm_view_rank3` matched all 32 values. This is a native
descriptor limitation **with a qualified historical workaround**, not a
blanket rank-3 kernel gap or a fresh board qualification of arbitrary ranks.
**Request:** preserve full shape/stride descriptors in a native load, or
document the supported decomposition conditions.

<a id="a5-up-031"></a>

### A5-UP-031 — Float immediates printed with six decimal places

Owner: **PyPTO Pro CCE code generation**. Evidence: **Historical + workaround**. Origin: survey 4b, D-121, D-136.

The prototype identified `std::to_string` in the ConstFloat printer
(`cce_codegen.cpp`, VisitExpr_): 0.004464285714285714 became 0.004464f,
1/sqrt(128) became 0.088388f, and sufficiently small positives became zero.
Compile-time ratios folded before printing and did not avoid the loss.
Ascriptor's `_imm_survives` detects loss and hoists exact runtime scalar
parameters, with bit-pattern register fallback for other dtypes. The
current adapter retains that workaround; no new compiler-binary probe here.
**Request:** round the double to FP32 once (ties to even), then print it
exactly, e.g. `%.8e` plus `f`; six-digit spellings that already read back
exactly may stay. Round-trip digits of the double (`%.17g`, max_digits10)
plus `f` are wrong: FP32 midpoint 1+2^-24 rounds up, not to even, and `2f`
is ill-formed. A local sweep matched single rounding for 2M doubles, 200k
midpoints and all subnormals. Existing kernels with the workaround are
not gaps. A candidate implementation of this request is kept as the
[float immediates patch](pypto-pro-supplements.md#4-exact-fp32-immediates-in-cce-code-generation);
Ascriptor does not apply it.

<a id="a5-up-032"></a>

### A5-UP-032 — Native VF register groups

Owner: **PyPTO Pro VF interface**. Evidence: **Source**. Origin: mapping register groups, D-230.

The fingerprinted `_vf_api.py` exposes no native register-count/group
parameter for load_align, astype, arange or arithmetic. The adapter's
`check_reg_groups` refuses reg<T,2> declarations and uses, including
load/store-only and cast-only bodies. Historical tests include all six
register-group samples and isolated uses.
**Request:** expose the native grouped carrier and matching lane extent.
Grouped predicate mapping remains **unmapped in Ascriptor**; that separate
implementation gap is not asserted to be an upstream absence by this row.

<a id="a5-up-033"></a>

### A5-UP-033 — Native integer VF floor remainder

Owner: **PyPTO Pro VF interface**. Evidence: **Source**. Origin: mapping vf.mod, D-230.

The inspected `_vf_api.py` has no native integer floor-remainder operation.
`vf.mod` is refused for single registers as well as grouped requests.
**Request:** expose an operation with floor-remainder semantics, including
negative operands, or specify a supported native equivalent. This is
separate from ordinary scalar `%` and from register-group support.

<a id="a5-up-034"></a>

### A5-UP-034 — Scalar access to storage-only low-precision dtypes

Owner: **PyPTO Pro**. Evidence: **Source**. Origin: new successor finding.

`block_ops.py::_check_scalar_supported_dtype`, called by both getval and
setval, rejects FP4/FP8/INT4/UINT4/HF4/HF8 containers as storage-only.
This is a **dtype-specific extension request**, not missing UB access:
ordinary supported-dtype GM/UB scalar access is connected and the two
FP32 UB examples pass all three backends.
**Request:** define a scalar access representation for these dtypes, if
supported, with explicit packed/bit semantics. Raw carrier access must
not silently be presented as an equivalent typed numeric scalar.

<a id="a5-up-035"></a>

### A5-UP-035 — PTO Mat TFILLPAD layout/value restrictions

Owner: **PTO ISA**. Evidence: **Source**. Origin: new successor finding.

The fingerprinted `npu/a5/TFillPad.hpp::TFILLPAD_IMPL` Mat overload requires
NZ layout and PadValue Zero/Null and performs zero fill. The current PTO
adapter supports positive zero; nonzero values and negative zero refuse.
**Request:** extend or document native Mat fill's supported pad values/layouts.
Positive-zero NZ fill is supported; this must not be reported as the
PyPTO alias/type compilation failure A5-UP-004. No new hardware claim is
made for arbitrary padding values.

<a id="a5-up-036"></a>

### A5-UP-036 — Loop-carried integer snapshots lost in native C++

Owner: **PyPTO Pro code generation**. Evidence: **Current native + workaround**.
Origin: M10-070, September 8–9; native evidence
re-acquired September 21.

On PyPTO **0.2.1, source build `fe20d72` / Ascend950PR (A5)**, whose native
launcher uses PyTorch 2.10.0+cpu and torch_npu 2.10.0, a kernel copies a current
integer cell into a previous cell before advancing the current one. Emitted
Python preserves that order, but native C++ commits the advanced loop argument
before reading its old value for another carry. A scalar snapshot chain returns
33 wrong values among 64; a three-cell rotation returns 37. Neither probe uses
attention arithmetic, UB, VF work or synchronization events. For the rotation the
generated C++ is three sequential loop-argument commits — `current = previous`,
`older = current`, `previous = older` — so the second and third read values the
first already overwrote. With the cell copies materialized it computes four
temporary values before any loop-carry commit.

[The native acquisition][carry-reacquired] ran both probes from library source
in the source investigation. The failing form is a negative control: the adaptation below
is switched off at run time and nothing else changes, so the two emitted kernels
differ only in the copied lines. It fails identically on a repeat run. With the
library as shipped both probes return all 64 values, and CCE and PTO ISA return
all 64 for the same source on the same box, so the DSL program is not at fault.
The receipt keeps the hashes of the four retrieved C++ files and of every
output. It covers these generic probes only.

Ascriptor's qualified adaptation materializes kernel Cube/Vector integer-cell
copies and `Var` initializers as `value + pl.get_block_idx() * 0`, after local
scalar folding. It preserves a real definition until native C++ has safe
temporaries; it adds no barrier or numeric approximation. **The upstream
generator was not changed.** This adaptation does not newly qualify floating,
boolean, VF or SIMT copies, every unsigned/mixed-dtype conversion, other hardware
or other upstream builds. A plain source identity that folds away earlier is
not an equivalent remedy; correctness adaptation is not a speedup claim.

**Request:** preserve the value-copy contract of
[RFC-0001 §5.2](rfc/0001-ir.md#52-mutable-scalars-are-cells-not-phis) when
eliminating loop arguments. Use temporary values or a correct parallel-copy
schedule, including cycles and initialized snapshots; alternatively expose an
explicit typed materialization operation that survives until this lowering is
safe. Keep the generic native controls as regression tests and report the exact
upstream build containing a repair. Until then, the recorded Ascriptor
adaptation is the qualified option for its stated domain.

<a id="a5-up-039"></a>
### A5-UP-039 — Acc-to-Mat insert selects the wrong mutex pipe

Owner: **PyPTO Pro**. Status: **open upstream, scoped kernel workaround**,
2026-09-16. The installed parser asks `get_op_pipe("insert")` without inspecting
memory spaces. Actual generated C++ places `get_buf`/`rls_buf(PIPE_MTE3, …, 0)`
around `TINSERT(Mat, Acc, …)`, whose data transfer uses FIX. Therefore these
locks do not wait for the Cube product before FIX reads L0C. `pl.move` cannot
substitute: its current public validator rejects Acc-to-Mat.

A one-chunk triangular-inverse probe passes the functional model but fails
native non-diagonal blocks. Adding an explicit M-to-FIX event before each
publish, while retaining FIX-to-MTE1 afterward, passes a five-chunk single-core
probe using both slots repeatedly. The three forward kernels retain this
explicit publication protocol. No vendor files or generated C++ are patched,
and no backend event planner is restored. This source snapshot contains no
complete-case hardware qualification for that change.
A general upstream fix must dispatch insert synchronization by its source and
destination memory spaces and qualify every supported data path.

The Delta Neumann variants now retain the same explicit M-to-FIX publication
edge. Their original complete cases and additional reused/odd/multi-core
controls were reported to pass in an earlier study. Rerun them on the target device
before making a current qualification claim.

<a id="a5-up-038"></a>
### A5-UP-038 — Native mutex pipe selection for Cube scalar fill

Owner: **PyPTO Pro**. Status: **fixed upstream**, commit `6a652e733` (2026-09-18);
first seen fixed in an inspected installation on 2026-09-16, which was not a claim
about every package carrying version 0.2.1.
Original evidence: **Source + Ascriptor guard**, 2026-09-15.
The tested installation is identified by the Python-tree and native-library
fingerprints in the IR-mutex receipt;
its 0.2.1 marker and the available source checkout are not an exact build identity.

`language/parser/_call_parser.py::_resolve_auto_mutex_pipe` sends `expands`
to `get_op_pipe`, which queries the CCE backend's registered `block.expands`
pipe. The inspected installation returns `PipeType.V`, including for a managed
Mat Tile used by a Cube scalar fill. That selects an unsupported Cube V-pipe
mutex. No new failing device execution is claimed for this source finding.

The repaired parser selects MTE2 for `expands` on Mat and retains V for Vec
(`language/parser/_call_parser.py::_resolve_auto_mutex_pipe`). Ascriptor carried a
portable source patch for it until 2026-09-22; with the baseline moved to upstream
`289942aa3` the patch, its guide section and the printer's supplement record are
removed, and the id is only tolerated in `ASCRIPTOR_PYPTO_SUPPLEMENTS`. An
installation older than `6a652e733` needs the upstream commit, not a local patch.
The adapter's old rejection and its fill-only bookkeeping are removed; the
original IR IDs, whole-tile/value checks and post-fill MTE2 barrier remain.
The L1 fill receipt identifies the exact package
fingerprints and generated native code. Four workloads (FP16/BF16 rotating
slots, a singleton with partial overwrite, and a UINT8 reinterpretation) pass
eight native runs, eight functional simulations and eight pipeline simulations.
The latter have no hazards or deadlock. The native C++ surrounds TEXPANDS with
mode-zero MTE2 mutexes. This closes this pipe-selection restriction on that
installation only; [A5-UP-004](#a5-up-004)'s separate `fillpad_l1` gap is unchanged.

<a id="a5-up-037"></a>
### A5-UP-037 — INT32 scatter casts UINT32 offsets to signed registers

**Backend/version:** PyPTO Pro source `bc593e0bcc5eeabd47ccea7d33463a53cf449e25`
(package marker 0.2.1), CANN 9.2.0 / `V100R001C12B056`, target `ascend950`.
Observed during the 2026-09-13 register TopK integration; this is separate
from the historical September 7 three-backend matrix.

**Reproduction:** an INT32 register of input identifiers is scattered to an
INT32 UB output with UINT32 lane offsets and a predicate. The generated PyPTO
source calls `vf.scatter` with those declared types. Its C++ backend first
requires UINT32 offsets for all 32-bit payloads, then chooses `int32_t` as
`idx_c_type` when the payload is INT32. `EmitVFScatter` at
`framework/src/interface/pypto_pro/backend/backend_cce_vf_ops.cpp:3494` contains
both decisions. The generated C++ therefore casts the offset register to
`RegTensor<int32_t>`, and this CANN compiler rejects both TopK index stores:
the signed-data `vscatter` overload requires a `vector_u32` third argument.
No device selection result was produced by the failing compile.

**Composite workaround:** [radix_topk](api/sorting.md#register-radix-selection)
carries the output buffer and index payload through UINT32 bit views. The
memory view is created in the caller so indexed tile-group identity is not
lost inside a VF callee. Address, payload bits and predicate are preserved;
the public index output remains INT32. This is explicit IR authoring, not a
numerical cast or a scalar search fallback. Validation belongs to the exact
source and dependency identities recorded for RFC-0014; no blanket repair of
native signed scatter or other integer widths is claimed.

**Request:** derive the scatter offset carrier from the required offset width
and signedness, independently of the data signedness; keep UINT32 offsets for
INT32 data. Retain a native signed-payload compile test on this CANN build.

<a id="a5-up-040"></a>
### A5-UP-040 — Documented `mrgsort2` argument order differs from its graph operands

Owner: **PyPTO Pro**. Evidence: **Current native + source**. Origin: p6-dma-mx import audit.

At Pro source `fe20d7268b2c8a09ec1188bb01ce0aecfeedc87c`, the documented call is
`pl.mrgsort2(src0, src1, dst, tmp, *srcs, exhausted=False)`
(`python/pypto_pro/language/_api.py:886`, API page `sorting/mrgsort2.md:24`), and
the doc test spells it that way (`python/tests/st/pypto_pro/frontend/docs/test_doc_sort.py:75`).
No parse handler is registered, so the default block handler keeps Python
argument order (`language/parser/_call_parser.py:1724-1732`). The IR registration
and CCE printer read `(dst, src0, tmp, src1[, src2, src3])`
(`framework/src/interface/ir/op/block_ops/sort.cpp:73-89`;
`framework/src/interface/pypto_pro/backend/backend_cce_block_out_ops.cpp:2265-2306`).
The documented spelling therefore merges `src1` with the `tmp` tile into `src0`,
using `dst` as scratch; the doc test checks no values. A historical board run of
that spelling dropped src0 and wrote 16 of 64 records (D-109,
[mapping](pypto-pro-mapping.md) `vec.mergesort_2seq`).

Ascriptor imports the graph order. A native A5 probe spelled `(dst, src0, tmp, src1)`
positionally merged both 16-record sources into `dst`, left them unchanged and left
a copy of the merge in `tmp`, consistent with the historical result. The import
maps that form to `vec.mergesort_2seq` and treats `tmp` as unspecified scratch.

**Request:** bind `mrgsort2` arguments by their documented roles, for example
in a registered parse handler, and give the doc test a numerical check.

<a id="a5-up-041"></a>
### A5-UP-041 — Attribute-free ops skip keyword validation the CCE emitter relies on

Owner: **PyPTO Pro**. Evidence: **Source**. Origin: p7 import profile review.

At Pro source `fe20d7268b2c8a09ec1188bb01ce0aecfeedc87c`, `OpRegistry::Create` checks call
keywords only when the op registers at least one attribute
(`framework/src/interface/ir/op_registry.cpp:116`). `vf.mem_bar` and `vf.store_unalign_post`
register none (`framework/src/interface/ir/op/vf_ops.cpp:589`, `:712`), so any keyword passes,
misspelled ones included. The CCE emitter still reads `mode` and `post_update` from the call
(`framework/src/interface/pypto_pro/backend/backend_cce_vf_ops.cpp:1010-1017`, `:3386-3388`), and
the `vf.load` / `vf.store` emitters read a `post_mode` string that neither the Python API nor
the registry declares (`:4058-4154`). Ascriptor's import profile pins the attributes it admits, so
compatibility profile `cann-pro-fe20d726-export/2` whitelists `mem_bar(mode)` and
`store_unalign_post(post_update)` by hand ([RFC-0015](rfc/0015-pypto-pro-import.md)); every other
keyword still fails at source, including the misspellings Pro accepts, and `post_mode` stays refused.
**Request:** register every keyword an emitter reads, validate keywords for every op, and
remove or document `post_mode`.

<a id="a5-up-042"></a>
### A5-UP-042 — Generated SIMT function names collide with libm overloads

Owner: **PyPTO Pro code generation**. Evidence: **Current native compile**. Origin: p6-simt-float import.

Pro prints a SIMT function under its Python name. A native compile-only probe on September 17
named them `remainder` and `fmod`: bisheng rejected `cce::async_invoke<remainder>(...)` and
`async_invoke<fmod>(...)` with "no matching function for call to 'async_invoke'", because the libm
overloads make the explicit template argument invalid. The same kernel with the function named
`rem_helper` compiled. Ascriptor's CCE printer reserves these names instead.
**Request:** rename or namespace generated functions whose names collide with C or C++
standard library declarations.

<a id="a5-up-043"></a>
### A5-UP-043 — FP16 `exp_sub` predicate documentation contradicts A5

Owner: **PyPTO Pro documentation**. Evidence: **Current native**. Origin: p6-fused-mask import, I011.

The API page `docs/zh/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/exp_sub.md:58`
states that for FP16 sources only even mask bits are valid. Native A5 probes found that `vexpdif`
tests the predicate at the source lane it reads, 2·i + layout, so with `layout=ONE` odd bits
select the results. Ascriptor models the measured rule.
**Request:** document the per-source-lane predicate for both layouts.

<a id="a5-up-044"></a>
### A5-UP-044 — `vf.load` / `vf.store` ignore their declared keywords

Owner: **PyPTO Pro code generation**. Evidence: **Source**. Origin: p6-vf-memory import.

The Python API documents `post_update`, `repeat_stride` and `count` for `vf.load` and
`post_update` and `repeat_stride` for `vf.store` (`python/pypto_pro/language/_vf_api.py:1810-1843`),
and the registry declares them (`vf_ops.cpp:829-847`). `EmitVFLoad` and `EmitVFStore`
(`backend_cce_vf_ops.cpp:4058-4154`) never read them: a strided load always prints
`vldus(..., POST_UPDATE)`, a store always prints `vstus` / `vstas` with `POST_UPDATE`, and
`count` or `repeat_stride` on a load changes nothing. Ascriptor refuses these keywords at source.
**Request:** implement the keywords or reject them.

<a id="a5-up-045"></a>
### A5-UP-045 — `syncthreads` under runtime control flow compiles despite its documented restriction

Owner: **PyPTO Pro**. Evidence: **Current native compile**. Origin: p6-simt import.

`pl.simt.syncthreads` must be reached by every thread and "cannot be placed inside runtime `if`,
`for`, or `while` control flow" (`python/pypto_pro/language/_simt_api.py:87`). The parser accepts
it there, and the CCE backend (`backend_cce_simt_ops.cpp:87`) prints `__sync_workitems();` inside
the branch or loop. September 17 compile-only probes placed it under `if (flag > 0)`,
`if (tid < 4)`, which only some threads reach, a `pl.range` loop and a `while` loop; all compiled.
A thread-dependent branch can leave the barrier unmatched at run time. Ascriptor imports only
top-level `syncthreads`.
**Request:** diagnose barriers under runtime control flow at parse time, or document the
supported uniform forms.

<a id="a5-up-046"></a>

### A5-UP-046 — `vf.full` of an FP4 register compiles to a `vbr` call with no FP4 overload

Owner: **PyPTO Pro**. Evidence: **Current native compile**. Origin: FP4 cast round-mode probe.

A vector function that declares a cast destination with `vf.full(0, dtype=pl.DT_FP4E1M2)` and then
assigns it from `vf.astype(..., dtype=pl.DT_FP4E1M2)` traces without a diagnostic. The generated CCE
source broadcasts the initial value with `vbr(reg, 0)`, and the vendor compiler rejects it: no `vbr`
overload takes a `RegTensor<float4_e1m2x2_t>` destination. On September 21 all five round modes of
the probe stopped there, before any kernel ran; the same kernel passes through the CCE and PTO ISA
backends with identical carriers (receipt). Ascriptor
printed that `vf.full` as the zero seed of a register whose view was declared ahead of the cast;
its register-init pass no longer seeds for a view's declaration, and the same kernel then passes
all five modes through PyPTO with CCE's carriers (receipt).
An FP4 register that is genuinely read before its first write still needs the seed and is still
not executable through PyPTO.
**Request:** reject an FP4 `vf.full` at trace time, or lower it to a form the compiler accepts.

<a id="a5-up-047"></a>

### A5-UP-047 — bisheng smashes its own stack selecting instructions for an MLA vector function

Owner: **CANN / bisheng**, with **PyPTO Pro** as the trigger. Evidence: **Current native compile**.
Origin: 2026-09-22 MLA board sweep.

`attention/a5_mla_fp16_bf16`'s `mla_online_m128` kernel does not compile on the box. The vendor
compiler terminates itself:

```
*** stack smashing detected ***: terminated
Running pass 'HiIPU VF DAG->DAG Pattern Instruction Selection' on function
  '@_Z27mla_online_m128_impl_vectorPU3AS1u6__bf16S0_S0_S0_S0_S0_f.vector.thread.7'
bisheng: error: clang frontend command failed due to signal
```

It is the compiler's own stack guard firing during instruction selection, not a diagnostic about
the source, so nothing in the printed program is named as wrong.

**The cce path compiles the same kernel with the same compiler.** This is not "bisheng cannot build
this kernel": the cce backend's output goes through `bisheng` / `ccec` too, and all three cases pass
end to end on the box that way. What differs is the C++ each path hands it. The crashing function is
`mla_online_m128_impl_vector...vector.thread.7` — PyPTO's own code generation for the vector
function — while ascriptor's cce printer turns the same IR into different C++ that compiles. So the
crash is bisheng's (a compiler must diagnose, not abort) and the trigger is PyPTO Pro's generated
form; either side fixing its half closes it.

It is **not** one kernel or one family. Three of this unit's 36 cases crash, across both of its
online kernels: `model_prefetch_query_wrap` and `model_prefetch_causal_boundary` on
`mla_online_m128`, and `model_paired_causal_lifetime` on `mla_online_paired`. Their siblings in the
same kernels compile and pass on the same cards in the same session — `model_prefetch_zero_idle`,
`model_paired_odd_partition`, `model_paired_even_partition`, `model_paired_tail` and
`model_paired_causal_decode` — so it is neither the box, nor the card, nor either vector function as
such, but what a particular scalar specialisation of one turns into. Reproduced on cards 7 and 6,
and all three have a cce run of the same case passing on the same box.

**What triggers it, from a one-variable experiment.** Across nine specialisations of these two
kernels the split is exact: the three that crash each contain a VF-internal floor **modulo** by a
constant with no quotient beside it, and the six that compile contain either a floor **division**
or no integer divmod at all. The modulus is not the variable — 257 and 67 are both prime, one
crashes and one does not; 192, 129 and 257 crash while 67 and 3 do not.

The experiment holds everything else still. The crashing case's own pushed bundle was copied
twice and run on the same card, same inputs, same delivery. Arm A is the emitted source
unchanged. Arm B rewrites only the two floor-modulo chains into the shape the passing cases
already emit — the same value, by floor division:

```python
# A (crashes):  r = cast(x) % C;  mod = r + (C if (r != 0) & (x < 0) else 0)
# B (compiles): q = cast(x) // C; r = cast(x) % C
#               mod = x - C * (q - (1 if (r != 0) & (x < 0) else 0))
```

A fails exactly as before. B prints `PYPTO_RUN_OK`. So the trigger is the emitted expression
shape: a remainder selected against a constant equal to its own divisor and added back, inside a
vector function. What the instruction selector then does with it is not observable from here — it
aborts on its own stack guard with no diagnostic.

That also means Ascriptor can route around it. `passes/integer_division.py` expands a floor
`scalar.mod` as `r + (b if adjust else 0)` and computes no quotient; emitting the quotient form
for a modulo reached inside a vector function would cost a few instructions and make these three
cases compile. That is a workaround, not the fix: the compiler should not be reachable this way.

This batch is not the cause, and that is measured rather than argued. For each crashing case, all
ten vector functions are **byte-identical** to the ones library `178f2dc` emitted, in both `manual`
and `auto_mutex` mode — the D-260 default change rewrote the scope body's credits and did not alter
one byte of the text the compiler crashed on. The unit was outside the pre-change board sweep, so
this is a first observation on this path, not a regression: its 2026-09-07 `pypto_pro` board row ran
library `3b37222` against an older installation.

**Worked around on our side, 2026-09-22, and the entry stays open.** First by refusing a floor
`scalar.mod` inside a vector function and rewriting the two MLA kernels through the quotient; then,
later the same day, by removing the trigger instead of the op. `pl`'s `//` and `%` have rounded
toward negative infinity since upstream `5866c9b6f` (2026-09-14), which is an ancestor of the
`289942aa3` minimum, so `integer_division` no longer expands a floor divmod for this backend at all
([mapping #34](pypto-pro-mapping.md)). The shape that aborts instruction selection is that
expansion, and nothing emits it now.

Measured, not argued: the three crashing cases were re-run **on their original source** — kernels
`a484f9c`, with `(first_row + row) % queries` unchanged — against a library that does not expand.
The emitted vector function carries a bare `pl.cast(...) % pl.const(192, ...)`, and all three
passed on the box. The refusal was removed with the trigger; an author who writes `x % n` in a vf
now gets a working program rather than a refusal, and the quotient spelling the two MLA kernels
carry is still correct, so it was left alone rather than costing a second evidence re-acquisition.

Nothing upstream changed. A kernel compiled by a PyPTO Pro that still emits the old expansion —
one built before `5866c9b6f` — would reach the same crash, which is why this entry stays open.

**Request:** a compiler that reports an error instead of smashing its own stack, and a fix for
the pattern the arm-A shape above reaches in `HiIPU VF DAG->DAG Pattern Instruction Selection`.
cce and PTO ISA are unaffected either way: `compact_integer_mod` keeps this op a single typed
helper call for them.

<a id="a5-up-048"></a>

### A5-UP-048 — `pl.load` refuses a zero-extent valid shape instead of moving nothing

Owner: **PyPTO Pro**. Evidence: **Current native run**. Origin: 2026-09-22 MLA board sweep.
Classification: **accepted restriction, decided 2026-09-22 — not a todo.** Kernels whose extent can
reach zero guard their own transfer; see [mapping #33](pypto-pro-mapping.md).

A core with no rows to do still reaches its load. The printed program narrows the destination to the
extent the kernel computed and asks for the transfer:

```
pl.set_validshape(slot_8, [0, 512])
pl.load(slot_8, view_5, [0, 0])
```

Upstream rejects it — `InvalidShape: load: offsets[0]=0 exceeds tensor dim 0 size 0`
(`block_ops.py:1186`) — so `attention/a5_mla_fp16_bf16`'s `model_paired_zero_idle` has no runnable
program. Offset 0 into an empty dimension addresses nothing and moves nothing; cce spells the same
IR as a zero-burst DMA and the interpreter as an empty slice, and both are no-ops. The check treats
an empty range as an out-of-range index.

The emission predates this batch: the same two zero-extent `set_validshape` lines are in library
`178f2dc`'s output under the old `auto_mutex` default, byte for byte.

**Decided, not requested.** The maintainer's call on 2026-09-22 was to treat this as an explicitly
unsupported form and to guard only the kernels that actually hit it, rather than wait for upstream
or sweep a guard through every idle-core case in the corpus. `a5_mla_fp16_bf16`'s
`online_paired.py` is the one kernel that did: its `if rows[context] > 0` guard costs nothing,
because `rows == 0` already takes the branch that zero-fills both destinations, so the skipped
transfer would have moved nothing. The guard changes 8 of that unit's 36 cases on **all three**
backends — cce and PTO ISA print the same skipped transfer and lose nothing — and the entry stays
here as the reason a kernel author will find when they write the next one.

## Supported paths and excluded classifications

- **UB GetValueFrom/SetValueTo:** both upstreams support ordinary same-dtype
  access. M10-061 repaired Ascriptor's adapters; the method/operator FP32
  examples pass all three backends. This is closed as an infrastructure
  defect, not listed as an upstream UB-target absence. A5-UP-034 is only the
  explicitly rejected storage-only dtype extension.
- **UB-to-L1 ND-to-NZ (prototype E1):** the blanket absence is superseded.
  `_ub_nd2nz` narrows an ND source to C0 and uses one pl.insert per NZ column
  alias. Later failures included our axis/arange handling and validshape
  restoration, not proof that the upstream composition was unavailable.
  M10-063 records the current descriptor repair. Unsupported geometry must
  be diagnosed separately rather than restoring the blanket claim.
- **L0C-to-L1 (prototype E2):** pl.move's whitelist is insufficient evidence
  of a missing operation. `pl.insert(mat, acc, [r,c])` is supported (D-116);
  current whole-row FP32/PTO repairs are in M10-063. A5-UP-007 retains only
  the partial-row FP32 limitation. Do not close that narrower issue from a
  passing whole-row result.
- **Ordinary strided VF offsets:** D-116 supports `base + offset` in aligned
  load/store. The historical cursor-walk-only claim is retracted. Unaligned
  cursor initialization remains the separate A5-UP-027 form.
- **Adapter TODOs and invalid requests:** an unmapped grouped predicate,
  an invalid immediate, or a barrier pair outside the twelve native MemType
  combinations is not independently established upstream missing support.
  Keep implementation/validation ownership until a valid native form is located.
- **Scalar versus VF/SIMT:** ordinary scalar abs/sqrt/cast, VF register
  abs/sqrt/casts, and SIMT scalar APIs are separate forms. Their success or
  failure cannot substitute for one another's validation.

## Prototype report reconciliation

P1-P6 and C1-C3 retain the original labels. R1-R10 name the ten Part 2 table
rows in their original order. E1-E3 name the three trailing bullets (the old
heading said "Two more", but contained three). This table is a provenance
crosswalk, not another list of independently counted gaps.

| Original row | Unified ID / disposition |
| --- | --- |
| P1 | [A5-UP-010](#a5-up-010): retain historical performance |
| P2 | [A5-UP-004](#a5-up-004): merge duplicate; current native compile evidence |
| P3 | [A5-UP-003](#a5-up-003): merge scalar cast; preserve separate historical induction-width cost |
| P4 | [A5-UP-011](#a5-up-011): retain performance; source confirms configuration writes |
| P5 | [A5-UP-012](#a5-up-012): retain build discrepancy; exact cause not isolated |
| P6 | [A5-UP-013](#a5-up-013): retain performance; source confirms split branch |
| R1 | [A5-UP-014](#a5-up-014): retain native i64/u64 VF arithmetic forms |
| R2 | [A5-UP-015](#a5-up-015): retain complex dtype absence |
| R3 | [A5-UP-006](#a5-up-006), [A5-UP-016](#a5-up-016), [A5-UP-017](#a5-up-017), [A5-UP-018](#a5-up-018), [A5-UP-019](#a5-up-019), [A5-UP-020](#a5-up-020): split FIX forms; partial offset workaround does not close native scale-loss defect |
| R4 | [A5-UP-005](#a5-up-005): merge duplicate native convolution absence |
| R5 | [A5-UP-021](#a5-up-021), [A5-UP-022](#a5-up-022), [A5-UP-023](#a5-up-023), [A5-UP-024](#a5-up-024): split conversion pairs; C2/C3 references deduplicated |
| R6 | [A5-UP-025](#a5-up-025): retain specific SIMT typing guards |
| R7 | [A5-UP-026](#a5-up-026): retain measured signed widening; distinguish conservative other-pair guard |
| R8 | [A5-UP-027](#a5-up-027): retain unaligned cursor; ordinary strided pointer offsets are supported |
| R9 | [A5-UP-028](#a5-up-028): retain arbitrary padding value surface gap |
| R10 | [A5-UP-029](#a5-up-029): retain innermost-stride gap |
| C1 | [A5-UP-023](#a5-up-023): retain historical wrong-lane evidence |
| C2 | [A5-UP-023](#a5-up-023): merge 64-bit conversion family; preserve distinct f32 compiler failure |
| C3 | [A5-UP-024](#a5-up-024): retain source-side packed i4 conversion |
| E1 | superseded blanket UB-to-L1 ND-to-NZ absence: C0 strip composition via pl.insert exists |
| E2 | superseded blanket Acc-to-Mat absence: pl.insert exists; partial FP32 remains A5-UP-007 |
| E3 | [A5-UP-030](#a5-up-030): native rank-3 descriptor limitation with qualified 2-D decomposition |

The source SHA-256 of the original report is
`14ad104b70654cd421aae15ae06eb034451da2e3f3430cda0a3157f8b758d623`,
from prototype revision `a147972cb52036df8ee9c206428339133ec9113d`.
Its original text remains in the imported successor Git history and the
read-only local source archive; neither is needed to use the current library.
Additional survey/mapping findings are included above as A5-UP-031/032/033;
the scalar storage-type and PTO Mat-fill source restrictions are A5-UP-034/035.
The later native loop-carry finding A5-UP-036 is not part of the prototype
crosswalk and does not change its historical issue or kernel counts.

[dependencies]: migration/fragments/a5-backend-dependencies-20260907.json
[scalar]: migration/fragments/a5-scalar-maintenance-20260907.json
[matrix]: ../../kernels/docs/migration/fragments/a5-three-backends-20260907.json "Historical kernel path; resolves at kernels f3b916d99f592687c6bbf3412148c163cd0c6875, not in the working tree"
[adapters]: validation/a5-backend-dependencies.json
[reconciliation]: migration/fragments/upstream-report-reconciliation-20260907.json
[carry-reacquired]: validation/m070-native-reacquisition.json
