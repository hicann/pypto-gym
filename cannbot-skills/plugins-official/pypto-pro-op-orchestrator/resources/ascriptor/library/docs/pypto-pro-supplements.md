# PyPTO-Pro dependency supplements

This is the recovery and application guide for the local PyPTO-Pro changes kept
by Ascriptor. It describes source changes to the dependency, not runtime monkey
patching or rewriting generated C++. A package's `0.2.1` version string does not
identify these fixes: inspect behavior and record file/native-library hashes.
The first two supplements are ones Ascriptor needs; the third is the candidate
implementation of an upstream request that Ascriptor itself works around.

| Supplement | Source patch | Evidence |
| --- | --- | --- |
| Integer scalar `pl.cast(value, dtype)` | Integer cast | Cast receipt |
| Read-only export observer and metadata reader | Export observer | [RFC-0015](rfc/0015-pypto-pro-import.md), `importers/pypto_pro/profile.json` |
| Exact FP32 immediates in CCE code generation | Float immediates | [A5-UP-031](upstream.md#a5-up-031) |

The patches are independent. Apply only a missing capability; an official release
that already passes the relevant probes does not need the corresponding patch.
The first changes Python only; the second also builds a native module, so it has
its own rebuild and re-pinning step.
These local dependency supplements do not qualify a kernel or device workload.

## Installed on purpose, never by default

Ascriptor applies none of these patches, and no installation step of its own implies
one: an unpatched `pypto_pro` stays unpatched. That is the point — the difference
between a box and upstream stays a decision rather than a side effect. What the
backend owes in return is to name the patch at the line that needs it, instead of
leaving the cost to PyPTO's own parser error several layers later. Supplement 1 is the
one the printer can need; supplement 2 belongs to the reverse direction, where export
already refuses without it, and supplement 3 Ascriptor never needs.

[`backends/pypto_pro/supplements.py`](../ascriptor/backends/pypto_pro/supplements.py)
owns the mapping from a printed form to a patch in this directory, and the probe below.
Each emission then

* warns once per supplement (`PyptoSupplementWarning`), naming the patch and this guide;
* repeats the list as a header comment of the generated `kernel_pypto.py`, so the source
  states its own dependency wherever it is read, copied or run by hand;
* records it in `manifest.json` under `pypto.supplements`, with the sites that needed it.

### Declaring one as already installed

`ASCRIPTOR_PYPTO_SUPPLEMENTS=integer-cast` — or `emit_module(...,
supplements=...)`, which replaces the environment rather than widening it — states that
the installation this source will run on already carries them. A declared supplement is
silent and skips its board-side probe, but stays in the manifest as `"declared": true`:
the record is what the run was made under, not a suppression of it. An unknown name is
refused rather than ignored, so a typo cannot quietly keep warning; a retired id is
accepted and ignored.

### The board-side probe

Emit knows what it printed; only the box knows what is installed, and the workstation
that emits generally has no `pypto_pro` at all. So the generated `run_case.py` carries
the probe and runs it before it compiles anything:

| Supplement | Probe against the installed package (every condition must hold) |
| --- | --- |
| `integer-cast` | `ir/op/block_ops.py::_ir_cast` has no rounding-mode default — the scalar dispatch cannot keep one — **and** `language/parser/_call_parser.py::CallParserMixin._VF_SCALAR_PL_OPS` contains `cast`, without which a vf body refuses the scalar form |

An absent supplement prints `MISSING PyPTO-Pro supplement: <id>` with the patch to apply,
then `PYPTO_RUN_SKIPPED`, and exits without selecting a device. An installation the probe
cannot read prints `PYPTO_SUPPLEMENT_UNKNOWN <id>` and the run continues: a probe that
cannot see is not evidence of absence.

## 1. Integer scalar cast

The patch is kept against upstream `289942aa3` (2026-09-21) and applies to it cleanly. It also applies
cleanly to `acabcaa780cc4cfdec7df91b2e9183d99bfb7fc2` (branch `9.2.0`, 2026-09-23), where it was
applied and installed; `git apply --check` is still the step that decides, not the revision.

### Contract and implementation

The unified API has two forms:

```python
value_i32 = pl.cast(value, pl.DT_INT32)             # returns a scalar
pl.cast(destination_tile, source_tile, mode=...)   # writes a Tile
```

Only the qualified `pl.cast` spelling is added for ordinary scalar code. The
existing Tile form retains `CAST_ROUND` when its mode is omitted. The scalar form
accepts signed/unsigned 8/16/32/64-bit integers and `INDEX` as sources, and the
eight ordinary integer dtypes as targets. It rejects a rounding mode, float,
bool, sub-byte, Tensor and pointer conversions. SIMT retains `pl.simt.cast` and
its separate conversion/rounding contract.

The scalar form creates the existing `ir.Cast` through `ir.cast`. Native codegen
already prints an integer conversion. This adds no allocation, buffer mutex,
kernel launch, C++ backend change, or host calculation. Conversions do not
saturate or check ranges. Preserve values by proving that they fit the target;
out-of-range signed conversions follow the target C++ toolchain rather than a
new portable numerical contract.

Changed implementation paths, relative to a PyPTO-Pro source checkout:

- `python/pypto_pro/language/_api.py`: declare/document both forms.
- `python/pypto_pro/ir/op/block_ops.py::_ir_cast`: dispatch Scalar + DataType to
  `ir.cast`, preserve Tile + Tile dispatch, and diagnose unsupported forms.

No separate scalar API name, imported bare-function alias, annotation rewrite,
or loop-index policy change is introduced. The patch includes parser, CCE
generation and native tests plus the public API documentation.

### Avoid re-widening an expression

`pl.range` still creates an `INDEX` variable in IR, although native VF codegen
uses a `uint16_t` induction variable. Plain Python integer literals also enter
IR as `INDEX`. An i32 annotation does not insert a conversion.

For the MLA-style expression, narrow the complete dividend and type the divisor:

```python
@pl.vector_function
def calculate(tile, first_row, rows):
    for row in pl.range(0, rows, 1):
        position = pl.cast(first_row + row, pl.DT_INT32)
        query = position // pl.const(67, pl.DT_INT32)
        # Use query in the original VF computation.
```

The caller must establish that every executed `first_row + row` fits i32.
Casting `first_row` alone is insufficient if adding the INDEX row immediately
widens it again. The same applies to modulo. This entry point does not change
the division semantics of PyPTO or Ascriptor. The Ascriptor
VF-width adapter now restores declared
narrow operands at VF div/mod and maps explicit integer casts. Other scalar
expressions still follow their existing promotion behavior. Its end-to-end
qualification is separate from the dependency-only receipt above.

### Validation to repeat

The patch supplies these tests in the dependency checkout:

```sh
python -m pytest -q \
  python/tests/ut/pypto_pro/language/parser/test_integer_cast.py \
  python/tests/ut/pypto_pro/codegen/test_cce_integer_cast.py
```

They check all 64 ordinary integer dtype pairs, INDEX/VF conversion, subsequent
type promotion, rejected forms, Tile mode/mutex compatibility, and Mat/Vec fill
pipe controls. Wider regression covers the parser, CCE generator and block/scalar
IR operation tests. Run the following on an available A5 device using the normal
environment/device selection and lock policy:

```sh
python -m pytest -q \
  python/tests/st/pypto_pro/frontend/vf_api/test_integer_cast.py
```

The native tests retain VF loops and integer division/remainder, include values
above the exact FP32-integer range, exercise i32 signed endpoints, and retain a
Tile-cast control. Numerical comparisons are exact. These are API probes, not a
fresh full-MLA qualification or a performance claim.

## 2. Read-only export observer and metadata reader

Import needs the complete attributes of every source Call, and Pro's own
`Call.kwargs` property cannot give them. Its binding in
`python/src/bindings/ir/ir.cpp` converts eight scalar types from the node's
`std::vector<std::pair<std::string, std::any>>` and has no `else`: an attribute of
any other type is dropped from the returned dict without an error or a marker.
The dropped types are the list and expression ones — `std::vector<int>`,
`std::vector<int64_t>`, `std::vector<std::string>`, `std::vector<ExprPtr>`,
`ExprPtr`, `MemorySpace` — which carry `mutex_ids`, `mutex_id_owner_indices` and
struct field names. Synchronization metadata is exactly what an
instruction-preserving import may not lose, so a Python-side workaround does not
exist; the values are only on the C++ side.

The patch adds two independent pieces:

- `KernelDef.parse_target_program(..., observer=...)`, a per-call read-only hook
  that receives the completed Program and its parser before parser-owned tables
  are released, and propagates exceptions to the caller. `export_kernel` checks
  for this parameter and refuses when it is absent.
- `pypto_pro._export_metadata`, a separate pybind11 module exposing
  `call_attributes`, `function_attributes` and `for_attributes` over the same
  `kwargs_`/`attrs_` vectors, converting fifteen types. An attribute type it does
  not know raises `py::type_error` rather than being skipped, which inverts the
  failure mode that motivated it. It transforms no IR, generates no code and uses
  no NPU. Pro's own property is left untouched.

Build the module into an isolated overlay, never into shared site-packages:

```sh
python <PyPTO-Pro-source>/tools/build_export_metadata.py --output-dir <overlay>/pypto_pro
```

The build links `libtile_fwk_interface` from the selected `pypto` native package
and sets `-D_GLIBCXX_USE_CXX11_ABI=0` to match that wheel's pybind internals, so
the result is specific to one Python, one native ABI and one machine. A different
Pro or native ABI needs a separately reviewed build, not a reused binary.

Record the built module's SHA-256 in `ascriptor/importers/pypto_pro/profile.json`
as `helper_sha256`. `prepare_import` requires every bundle's recorded
`native_helper` to equal it, so a rebuild that does not reproduce the byte-identical
module — which an optimizing C++ build generally will not — requires revising the
profile. That is the expected path, not a fault: the previous profile becomes a
registered predecessor and `session.same_export` keeps accepting exports recorded
under it.

## 3. Exact FP32 immediates in CCE code generation

Unlike the first three, Ascriptor does not need this patch. It is the candidate
implementation of the [A5-UP-031](upstream.md#a5-up-031) request, kept so the
request has a working reference rather than only a specification.

Pro's `ConstFloat` printer spelled immediates with `std::to_string`, which gives
six decimal places: `0.004464285714285714` printed as `0.004464f`, `1/sqrt(128)`
as `0.088388f`, and small positives as zero. Folding a ratio at compile time did
not avoid the loss. The patch adds `CceFloatLiteral(double, Span)` in
`framework/src/interface/pypto_pro/codegen/codegen_base.cpp`:

- Round the double to FP32 once, round to nearest with ties to even.
- Keep the legacy `std::to_string` spelling byte for byte where it already denotes
  that FP32 value, so existing generated code does not churn.
- Otherwise print nine significant digits (`%.8e`), which a correctly rounding
  compiler reads back as the same FP32 value.
- Keep the builtin spellings for NaN and infinities, and raise `ir::ValueError`
  when a finite value would round to infinity in FP32.

Nine digits rather than the double's round-trip digits is deliberate: `%.17g`
plus `f` is wrong, because the FP32 midpoint `1+2^-24` rounds up instead of to
even, and a spelling like `2f` is ill-formed.

This one changes C++, so it needs a full native rebuild, not the Python-only
installation the other supplements use. Ascriptor's own path is unaffected either
way: `ascriptor/backends/pypto_pro/emit.py::_imm_survives` detects a lossy spelling
and hoists an exact runtime scalar parameter, with a bit-pattern register fallback
for other dtypes. Kernels carrying that workaround are not gaps and need no change
if the patch is ever applied upstream.

## Applying to an official checkout

1. Identify the imported `pypto_pro` package, source revision and native library
   hashes. Probe the capabilities first; do not infer presence from a version
   string, a similarly named API, or a source checkout that was not installed.
2. Review the relevant patch against the actual official source. With `PYPTO_SRC`
   and `ASCRIPTOR_SRC` set to the respective checkout roots, check it first:

   ```sh
   git -C "$PYPTO_SRC" apply --check \
     "$ASCRIPTOR_SRC/docs/patches/pypto-pro-integer-cast.patch"
   git -C "$PYPTO_SRC" apply \
     "$ASCRIPTOR_SRC/docs/patches/pypto-pro-integer-cast.patch"
   ```

   If context has changed, port the small semantic change to the named owner
   function rather than overwriting unrelated upstream files. Keep existing fixes
   intact.
3. Install the changed Python package through that environment's normal source
   installation process. Supplement 1 requires no native-library rebuild when the
   existing `ir.cast` support is present; record the changed Python hashes and the
   unchanged native hashes. Supplement 2 builds one separate module into an isolated
   overlay and re-pins `helper_sha256`, and supplement 3 changes code generation and
   needs a full native rebuild, so both change native hashes too. Verify imports in a
   fresh process, and retain the previous installation or source revision for
   rollback. Then declare what this installation now carries, so its emissions stop
   reporting it as missing: `ASCRIPTOR_PYPTO_SUPPLEMENTS=integer-cast`.
4. Run the focused, compatibility and native checks above. Use generated code to
   distinguish a real i32 dividend from a cast that was subsequently widened.
   Preserve failure artifacts and stop qualification if the original numerical
   or synchronization gates fail.

The local source candidate was based on `d132c31` and changed only the two Python
implementation files named above. Tests used an isolated overlay of the actual
installed package, retaining unrelated installed parser/runtime differences.
The portable patches and receipts are the retained recovery artifacts; private
machine settings, package backups and raw logs remain in ignored task storage.
