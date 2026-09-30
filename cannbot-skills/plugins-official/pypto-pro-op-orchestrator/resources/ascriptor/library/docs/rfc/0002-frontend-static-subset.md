# RFC-0002: The frontend — AST compilation and the static subset

Status: implemented for the vertical slice in M2 (`ascriptor/frontend/`); M3 extends it to the whole corpus. §8 lists what exists.
Depends on: RFC-0001; decisions D-004, D-008, D-014.

## 1. Principle

A decorated kernel function is **never executed**. Its AST is compiled into Surface IR. Every
sub-tree is classified as **static** (its value is decided by Python at compile time) or
**dynamic** (its value is an IR value or a handle to one). Static sub-trees are evaluated by
the Python interpreter, in the function's own globals and closure, with no restriction on what
they call (decision: arbitrary Python calls with all-static arguments are allowed); dynamic
sub-trees are compiled into ops. This is the Triton `constexpr` rule without the annotation:
taint decides.

Why an AST compiler rather than the old trace: the compiler sees names, so shape symbols
become parameters (no `shape_bindings`), native `if` / `break` / `continue` / `and` / `or`
compile directly (no source rewriter), rebinding mistakes are diagnosable, and a kernel is
compiled once per device instead of re-traced per call.

## 2. Surface kept from easyasc

The DSL surface is unchanged except for the approved breaking list (D-008): `@kernel`, `@vf`,
`@simt`, `Tensor`, `GMTensor` / `GMTensorList`, `Var`, `TBuff`, `Reg`, masks, events,
`<<=` for copies, the same instruction helpers and the same CamelCase helper names
(`GetCubeIdx` etc.). Imports come from a device facade: `from ascriptor.a5 import *`.
Importing a facade binds names only; it never mutates process state. A kernel is compiled for
the facade it was written against; the same source may be compiled for another device by
importing another facade in another module.

## 3. Signatures

### 3.1 Parameter order and outputs

Parameters are `inputs → outputs → scalars`, as before. Outputs are the parameters the kernel
**returns** (`return z` or `return z, w`), as before; the frontend checks that every returned
name is a parameter and records them in `outputs`. A parameter that is written but not
returned produces a warning naming the write.

Resolved annotation objects are taken from the Python function's `__annotations__`; they are
not evaluated a second time. This preserves locally imported GM/dtype names used only in
annotations, which Python does not retain as closure cells. Postponed annotations retain their
source-AST resolution from the function's available globals/closure (M10-011).

### 3.2 GM tensors with symbolic shapes

```python
@kernel
def matmul(x: GM[f32, ("M", "K")], y: GM[f32, ("K", "N")], z: GM[f32, ("M", "N")]):
    ...
    return z
```

A dimension in a signature is an integer literal, a **string symbol** (D-014), or a product
of these written as a string (`"M*K"`; multiplication only). Each symbol becomes an implicit
`i32` scalar parameter after the explicit scalars, in first-appearance order; the launcher
derives its value from the real tensor and checks that every occurrence agrees. The kernel
body reads a symbol as the scalar it is (`x.shape[0]` is the value `%M`). An explicit scalar
parameter with the same name as a symbol is the same parameter (this is how the old
`(x, y, z, M, N, K)` call shape is kept: the caller may pass the value, the launcher checks it).

Bare `x: GMTensor` is retained only as a diagnostic marker and is rejected by the compiler.
Use explicit `GM[dtype, dimensions]` annotations; rank and dtype are compiler inputs, not inferred
by a hidden launch trace. `shape_bindings` and `LITERAL_SHAPE_DIM` are gone. This corrects the
historical prose against the implemented signature contract during M10 declaration review.

Views (`reshape`, `flatten`, `T`, slicing) are written in the body only, never in a signature.

### 3.3 GM tensor lists

```python
def gather(xs: GMList[bf16, ("?", "D")], out: GM[bf16, ("N", "D")]):
    for i in range(xs.count):           # dynamic: cf.for over list.count
        t = xs[i]                       # gm<bf16, [?, D]>; t.shape[0] is a runtime scalar
        rows = t.shape[0]
        ...
```

`?` marks a per-member dimension read from the list descriptor at runtime; shared dims are
symbols checked by the launcher across members. `len(xs)` / `xs.count` is dynamic unless the
launcher pins it (`GMList[bf16, ("?", "D"), 8]` makes it a literal and unrolls). Members
may differ in length (D-014). Implemented 2026-08-28 (D-048): `xs.count` / `len(xs)`, `xs[i]`,
`for t in xs` (a device loop over `list.count`, unrolled when the count is pinned), and
`t.shape[0]` of a member as the `list.item_dim` scalar; sample `tests/kernels/a5/samples/list_concat.py`.

### 3.4 Scalars and `block_dim`

Scalar parameters are annotated with a dtype (`n: i32`, `alpha: f32`; the old `Var`
annotation means `i32`). `@kernel(block_dim=...)` takes an integer, a symbol, or a string
expression over symbols in the grammar `int | sym | expr ('+'|'-'|'*'|'//') expr |
ceil_div(expr, expr) | min(expr, expr) | max(expr, expr)`; the launcher evaluates it. The
default is the device's cube-core count for `mix` / `cube` kernels and the vec-core count for
`vec` kernels, as today.

### 3.5 ACLNN-portable parameter names

GM input and scalar parameter names must remain unique after CANN's ACLNN API lower-camel conversion:
the first character is lowercased and an underscore capitalizes the next component
(`B` -> `b`, `BH` -> `bH`, `snake_name` -> `snakeName`). The frontend reports E0011
at the later declaration when two explicit parameters, or an explicit parameter and an
implicit shape scalar, map to the same API name. Output arguments follow a separate CANN
decoration rule and are not part of this input/attribute collision check.

The restriction does not apply to local names or to `@vf` / `@simt` parameters: those do
not enter the OpDef/ACLNN host API, and CCE/C++ keeps them case-sensitive. Thus a local
tensor `b` and a local `Var` named `B` are valid; a GM input parameter `b` and scalar parameter
`B` are not. This front-end fence replaces the late generated-C++ redeclaration diagnosed
in M10-080.

## 4. Static evaluation

### 4.1 Environment

The compile-time environment is the kernel's `__globals__`, its closure cells and default
arguments, plus a local scope. Parameters are dynamic. A local name is bound to either a
Python object (static) or an IR value / handle (dynamic). Containers (`tuple`, `list`,
`dict`) are static structure that may hold dynamic elements; indexing them needs a static
index.

### 4.2 Expressions

| node | rule |
|---|---|
| constant, static name | static |
| dynamic name | dynamic |
| operator on static operands | evaluated by Python |
| operator with any dynamic operand | compiled: scalar ops for scalar-like, the desugar table (RFC-0001 §12) for tensor / register operands |
| `and` / `or` / `not` with a dynamic operand | `scalar.and` / `or` / `not` on `b1` values (no short-circuit; both sides are evaluated — the old bitwise rewrite made the same choice) |
| comparison with a dynamic operand | `scalar.cmp.*` |
| call: static callee, all-static args | called by Python; result must be a static-representable value (int, float, bool, str, None, dtype, tuple / list / dict of these, or a DSL descriptor such as a type); recorded in `module.static` with a digest |
| call: DSL primitive or method on a dynamic value | lowered by the primitive's rule |
| call: Python function with a dynamic argument | **inlined**: its source is fetched and compiled in the callee's own globals with parameters bound; recursion depth limited (default 32); `lambda` with a dynamic argument is unsupported in v1 |
| subscript on dynamic tensor | `mem.view` / `mem.slice` (static or dynamic offsets, static or dynamic extents) |
| attribute on dynamic value | DSL property (`.shape`, `.dtype`, `.T`) or method |
| comprehension | iterable must be static; unrolled |
| f-string with a dynamic part | error: use `debug.print` |
| `static(expr)` | forces static evaluation; error if `expr` is dynamic |

### 4.3 Statements

| node | rule |
|---|---|
| `x = expr` | binds `x` in the local scope (static or dynamic). Binding a name that currently holds a **cell** to a new dynamic value is an error (E0102) suggesting `x.set(...)`; the old trace silently rebound. |
| `x += expr` | cell → `scalar.set`; static → Python; other dynamic → `x = x + expr` |
| tuple unpacking | target count must match a static container |
| `for` over `unroll(...)` or a static container (list, `enumerate`, `zip`, …) | unrolled; the loop variable is static in each copy; a static `break` stops unrolling; `unroll` needs static bounds |
| `for i in range(...)` (static or dynamic bounds) | `cf.for`; bounds and step are integer scalars, with one to three arguments and nonzero step; `i` is a dynamic `i32`; a constant-bound loop stays a loop in the generated code (D-027) |
| `for t in xs` with `xs: GMList` | `cf.for` over `list.count` with `list.item` |
| `if` with static test | the untaken branch is not compiled |
| `if` with dynamic test | `cf.if`; both branches compiled; a dynamic name bound inside a branch or loop body is unavailable after it (E0111 on use, naming the region and suggesting a `Var`); rebinding an outer dynamic name inside is E0110 |
| `while` | unsupported (E0201) |
| `with autosync():`, `with cube_scope():`, `with vec_scope():` | `region.*` |
| `assert` | static test → checked now; dynamic → `debug.assert` (sim-only; codegen backends drop it with a note) |
| `return` | kernel: last statement, outputs only; inlined helper: at most one, last |
| `break` / `continue` | in `cf.for` → `cf.break` / `cf.continue`; dynamic condition inside an unrolled loop → E0120 |
| `try`, `with` (other), `global`, `nonlocal`, `yield`, `async`, `del`, `match` on a dynamic subject, `*args` / `**kwargs` with dynamic content | unsupported, each with its own code |
| `static_print(...)`, `static_assert(cond, msg)` | compile-time print / check |

Integer loop counts must remain integer expressions. In particular, `Var(32 * 128 / 64)`
holds a float because its initializer is a static Python expression; use integer division
for an integral trip count. Float bounds are rejected at their source location rather than
coerced in a backend. C310 VF loops require an integer comparison; a float bound previously
escaped to the vendor compiler and triggered its `ICmpInst is required` assertion (M10-009).

### 4.4 Determinism and side effects

Static calls may run arbitrary Python; the compiler does not sandbox them. Their results are
part of the IR and their `(loc, qualname, digest)` are recorded so two compilations can be
compared. A static call with side effects is the author's responsibility, as it was under trace.

### 4.5 Scoping

Python's function scope applies, restricted as above: a name bound inside a `cf.for` or
`cf.if` body to a dynamic value is not visible after the block (the IR value would not
dominate). A static value bound inside a static (unrolled) loop is visible after it, as in
Python. Shadowing a DSL builtin (`range`, `Tensor`, …) is a warning.

## 5. Diagnostics

Format: `path:line:col: error[E0102]: message`, followed by the source line with a caret and,
when relevant, a note with the static context (`note: 'n' was bound here as static int 32`).
Codes: `E00xx` names and scope, `E01xx` binding and dataflow, `E02xx` unsupported syntax,
`E03xx` types and shapes, `E04xx` device / capability, `E05xx` static evaluation failures
(the Python exception is chained). Every diagnostic has a stable code and one test.

`E0011` is the ACLNN-portable kernel-parameter-name collision from §3.5.

## 6. Compilation artifacts and API

`kernel.ir()` returns the Surface module. `ascriptor.runtime.compile_kernel(kernel, backend=...)`
returns emitted artifacts; the CLI provides `dump-ir`, `check`, `compile` and `explain`.
There are no public `kernel.lowered`, `kernel.emit` or `kernel.explain` instance methods.
Maintenance code may use `PassManager(PIPELINE).run(module)` to inspect Lowered IR, under the
internal compatibility boundary of RFC-0012.

## 8. Status after M2

| piece | file | notes |
|---|---|---|
| DSL surface (markers) | `frontend/dsl.py`, facade `ascriptor/a5.py` | old names kept: `DT.float`, `Position.L1`, `Tensor`, `DBuff`, `Var`, `Reg`, `MaskReg`, `VcMutex`, `auto_sync`, `matmul`, `cast`, `CeilDiv`, `GetCubeIdx`, … plus `GM[dtype, dims]`, `GMList`, `f32` / `i32` annotation names |
| compiler | `frontend/compiler.py` | taint-driven: static sub-trees go to Python (`static_eval`), dynamic ones become ops; `Dyn` wraps IR values, the IR type decides operator / method meaning |
| signatures | §3 | `GM[f32, ("M", "K")]` symbols become the scalar parameters of the same name (explicit or implicit); order inputs → outputs → scalars checked; outputs = the returned parameters |
| statements | §4.3 | `=` (rebinding), `<<=` copies, `+=` on cells, `for … in range()` → `cf.for`, `for … in unroll()` / static containers → unrolled, static and dynamic `if`, `with auto_sync() / vec_scope() / cube_scope()`, `assert`, `break` / `continue`, `return` |
| calls | §4.2 | markers → ops; `@vf` / `@simt` callees compiled once per call-site type signature, tensor dims that name kernel values become callee parameters, read / write sets from the callee's access sets; plain Python helpers with dynamic args are inlined; static calls run in Python |
| values named after Python variables | `bind()` | `l1x = Tensor(...)` produces `%l1x`; temporaries take the op name (`%mul`, `%abs`) |
| diagnostics | `frontend/errors.py` | `path:line:col: error[E0xxx]: message` with the source line; codes per §5 |
| tests | `tests/frontend/`, `tests/test_vertical_slice.py` | the five kernels compile, verify, are pinned as goldens (`tests/goldens/ir/compiled/`), and reproduce the functional goldens on the reference interpreter bit for bit |

Not yet after M2 (all done in M3, see §9): `reshape` / `flatten` / `.T` views, the remaining register methods.

## 9. Status after M3

The frontend covers the whole a5 corpus (D-024). The compiler is split by what it compiles:

| piece | file | notes |
|---|---|---|
| compile-time values | `frontend/values.py` | `Dyn` (an IR value, optionally with *riders*: `.T`, `.relu()`, `.requant()`, `.subblk()`, `.single()` …), `ElemOffset` (`t[k]`), `RegExpr` (a deferred register expression, the old `RegOP`), `RegList`, `Img2col` |
| statements, expressions, calls, static evaluation | `frontend/compiler.py` | unchanged model; adds tuple loop targets, `*args` from static containers, comprehensions over static containers, `zip` / `enumerate` / `len` on dynamic values, plain helpers that build DSL objects (`def _cvmutex(): return CvMutex(...)`) compiled inline |
| registers | `frontend/rules_reg.py` | `Reg` / `RegList` / `MaskReg` construction; methods and operators build `RegExpr` trees that are emitted when they meet `<<=` (inner links into temporaries in source order, the outermost op into the target, RegList reductions as the old pairwise tree); loads and stores pick the LoadAlign / StoreAlign distribution from the tensor rider and the element sizes; every stub form (`ub_to_reg_normal(reg, ub)`, `expsub(dst, a, b)`, `vf_barrier(...)` …) |
| tensors | `frontend/rules_mem.py` | geometry (`shape / offset / span / sliced dims`) kept per value so explicit DMA stubs infer their parameters as the old `cube.py` / `vec/datamove.py` did; GM views (`gm[i, a:b, :]`, `reshape`, `flatten`, `reinterpret` with C0 scaling, `.nz()`); `<<=` between memories compiles to `dma.copy` with the riders as attributes (`transpose`, `relu`, `scale`, `offset`, `hif8_hybrid`, `dual_mode`, `sub_block_id`, `atomic`); explicit instructions compile to the specific ops (now legal in Surface IR); `matmul` / `matmul_mx` / `conv2d` stay macro ops |
| synchronisation and the rest | `frontend/rules_sync.py` | events (`SEvent` … `QEvent` → `sync.event` typed `event<depth, set, wait>`), `setflag` / `waitflag`, `bar_*`, cross-core signals with their default pipes, `with atomic_add():`, `Var.GetValueFrom` / `SetValueTo`, `kernel_print` / `kernel_dump_tensor`, `split_workspace` |
| DSL surface | `frontend/dsl.py` | every name the corpus uses, with the old spellings (`RoundMode.AWAY_FROM_ZERO`, `RegLayout.ZERO`, `MaskType.LOWEST32`, `DualMode.SPLITM`, `VfPipe.STORE` …); the host dtype codecs (`fp32_to_hif8`, `FP4_E1M2_MAX_VALUE` …) re-exported from `ascriptor.dtypes` |
| porting | `tools/port_kernel.py` | mechanical, source-preserving port of an old script: facade import, `GM[dtype, dims]` signatures synthesised from the recorded golden exactly as the old `OpExec` inferred the GM shape, `range(name=)`, host code dropped; hand edits are listed in the file header |
| gate | `tests/kernels/a5/corpus.json`, `tests/test_corpus_a5.py`, `tools/corpus_status.py` | every ported kernel compiles and verifies or is excluded with a reason; `corpus_status --replay` reports the bit-exact replays on the reference interpreter |

Semantics decided while porting: an int literal into a float register converts (the old DSL raised);
a converting register store whose size pair the old DSL silently dropped is an error; `astype(dtype)`
still yields the target's dtype on a direct `<<=` (the old rule); `unroll()` unrolls and `range()` always loops (D-027); `RegList` indices must be static; the facade switch `if os.environ[...]: from easyasc.a5
import *` of a few scripts is ported as its default branch (`a5pr`).

Not yet: complex-valued register immediates (the two complex kernels are excluded), `lambda`
with dynamic arguments, `while` outside simt, `GMList` iteration and `?` dims (M5).

Replay on the reference interpreter (`docs/corpus_a5.md`, generated from
`tools/corpus_status.py --replay`): of the 133 kernels in the 79 scripts (the 129 old ones plus
the four dtype-specific kernels split out of `matmul_quant.py`, D-026), 120 compile and verify
(the other 13 are the documented exclusions); 117 of those have recordings and 112 replay every
recorded case bit for bit, with outputs poisoned at launch — no kernel of the corpus leaves an
element unwritten. The tracked set (111 cases of 91 kernels, RFC-0003 §7) replays completely
except the three `replay_xfail` entries: the two complex-register kernels and the single-core
flash-decoding schedule of `v8_allhif8`. Two untracked kernels (`pfa_fd_v6_allhif8`,
`v7_allhif8`) differ in one byte (a 1-ULP hif8 rounding of the old micro pipeline); every other
recorded kernel, the large untracked cases included (the `gdn_legacy` bridges, `recompute_wu`,
the `mha_ifa` family), replays bit for bit under `corpus_status --replay tmp/goldens_full`.
Getting there took the
interpreter's M3 extensions listed in `backends/sim/interp.py`'s docstring, and one frontend fix
found by the replays: a `RegList` load or store advances by one register of lanes per element
(the element size of the *register*, not of the tensor), which the old kernel
`matmul_kmkn_blockwise_quant128` exposed on a f32 → e5m2 pack4 store.

## 7. Open questions

1. Whether `lambda` with dynamic arguments should be supported by compiling the lambda's AST
   (source retrieval via `inspect` is fragile on the same line). Default: unsupported in v1.
2. Whether a dynamic `and` / `or` should short-circuit via `cf.if`. Default: no (matches the
   old behaviour); document that both sides are evaluated.

## 10. M10 declaration review corrections

`mulscast(dst, src, value, mask=None, *, layout=RegLayout.ZERO)` takes its predicate as the
fourth positional argument or `mask=`. Inactive source lanes produce zero in the expanded
destination layout, as specified by the VF model; the frontend must not drop a positional mask.
`print_reg(reg, label="", lanes=8)` emits `debug.print_reg` inside a VF, with a static label and
nonnegative integer lane count. Source generation can omit this simulator observation with its
documented note; it is not a hardware printf promise.

An explicit launch override takes precedence over a decorator block_dim. The simulator must
honor the decorator's integer/expression after shape/scalar binding, just as the launch contract
requires, and reject nonpositive counts. It must not silently expand a one-core decorator to all
device cores. The expression grammar is the restricted arithmetic/call grammar in §3.4.

An L1 source `.T` is valid for both L0A and L0B copies. The frontend carries the transpose
into the existing `dma.copy` attribute, and device lowering selects `dma.l1_to_l0` with
`src_is_transpose`. The L0 geometry retains the source span and transpose provenance for MMAD
dimension inference. Rejecting this path before the existing L0 handling was a frontend defect.
