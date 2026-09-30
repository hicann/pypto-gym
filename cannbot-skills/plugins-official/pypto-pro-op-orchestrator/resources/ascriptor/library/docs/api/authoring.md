# Typed entries, scalar state and source control flow

Use `ascriptor.a5` for A5 authoring. A facade binds the decorated entry's
device without changing another imported facade. A5 exposes `vf` and `simt`.
`.a2` and `.a3` provide their own declared forms; qualify the selected workload
and the unit contracts; the separate `.a5pr` facade remains experimental.

```python
from ascriptor.a5 import GM, Var, i32, kernel


@kernel(mode="vec", block_dim=1)
def total(o: GM[i32, (1, 1)], count: i32):
    value = Var(3)
    for index in range(count):
        value += index
    value.SetValueTo(o)
    return o
```

`GM[dtype, shape]` declares each argument's logical dtype and shape. Repeated
dimension symbols share one launch binding. The caller provides output storage;
the returned tensor identifies observable output. See [the runnable scalar
entry](../../examples/api/scalar_math), [typed AXPB
entry](../../examples/api/axpb) and [GM-list
example](../../examples/api/list_concat). Legacy bare `GMTensor`
annotations do not provide enough information for a supported signature.

`kernel` supports `@kernel`, `@kernel()` and explicit `mode="mix"|"vec"|"cube"`
and `block_dim`, and takes nothing else — there is no `name=`. It returns an entry
inspected with `.ir()` and executed through `OpExec` or another launcher. Direct Python
invocation raises a diagnostic. Uppercase characters in the function name are accepted;
the old blanket naming restriction does not apply.

**The entry's name is the decorated function's name**, and it is what the emitted
sources, the diagnostics and the manifest call it. Two variants of one kernel therefore
need two differently named functions; a factory that returns the same decorated function
under another binding leaves both carrying the original name.

`func` inlines a reusable helper in decorated source. `vf` defines a register
function called from a kernel with UB tensors and scalar arguments. `simt`
defines a thread-parallel function; allowed thread counts are 64, 128, 256, 512,
1024 and 2048. Its argument/body restrictions and atomic examples are described
in [the SIMT units](../../examples/api/simt_atomics). A valid decorator
does not by itself establish that a body or backend supports an operation.

The A5 [radix_topk composite](sorting.md#register-radix-selection) uses this
same inline/VF machinery after validating its public call contract. Its
algorithm body remains visible in IR and uses the existing backend and model
paths; callers do not need to copy the implementation.

## Loops, branches and scopes

The frontend compiles the function's source AST. Native `if`, `elif`, `else`,
`break` and `continue` express device control flow. `range` remains an ordinary
Python built-in in host reference code. In decorated source, `range` produces
a runtime loop; `unroll` copies a body for static iteration values. A loop
accepts one to three integer bounds, no keywords, and a nonzero step. Use `//`
for integral static trip counts: `range(64 / 16)` is rejected even though its
mathematical result is an integer. Dynamic bounds belong in `range`, not
`unroll`. [Buffer ring](../../examples/api/buffer_ring) exercises an A5
runtime loop. The retained A2 `scalar_control.py` entry covers empty loops, continuation and early exit
at its declared A2-only scope.

`vec_scope` and `cube_scope` route operations to one side. They do not order
memory accesses. `with auto_sync():` is a compiler region that orders supported
same-side dependencies; it is not a Python decorator and does not establish
cross-side ownership. [The explicit event rings](../../examples/api/event_depths)
and [A5 cube/vector roundtrip](../../examples/api/cube_vector_roundtrip) show separate
ownership mechanisms. Legacy `If`/`Elif`/`Else`, `break_`/`continue_` and an exported
`range(..., name=...)` are not current facade APIs.

## Scalar values and memory

`Var(value, dtype=None, name="")` creates a mutable device scalar cell. Use an
explicit dtype for widths, signedness and values that feed instruction operands.
The constructor and scalar operations are compiled; the old host object fields
`.value`, `.idx` and `Expr` are not public authoring interfaces. `Var(existing)`
creates another cell. Use `.set(...)`, augmented arithmetic or the scalar-memory
operators below to update a cell. Plain `=` binds a name at compile time and
does not write into the existing cell; rebinding a `Var` to a dynamic scalar
expression is rejected with `E_REBIND_CELL`.

`GetValueFrom` and `SetValueTo` load/store one element at the beginning of a
selected tensor view. They are kernel-level operations, not reductions or VF
register movement. GM, workspace and UB scalar memory are supported storage
classes; L1 and L0 are rejected.

Kernel level means the kernel body or an inlined `func`. The same call inside a
`vf` or `simt` body is rejected with `GetValueFrom is a kernel-level operation`;
either `vec_scope` or `cube_scope` accepts it. The element is the first of the
supplied view, addressed flat over the root's declared shape, and a wider view
loads that first element with no diagnostic at all — slice down to the element
itself, `x[row, col : col + 1]`, where both subscripts may be dynamic. That is
the indexed per-row read: compute the index at kernel level, then copy the
selected row with an ordinary [slice](storage.md#indexing-and-slicing).
[Indexed row gather](../../examples/api/indexed_row_gather) runs that
read once per slot, gathers two tables at their own row widths, and records what
an unconditional copy and a clamped destination row each cost. A memory
dtype that differs from the cell's inserts a scalar cast rather than an error,
so an FP16 `3.25` reaches an `i32` cell as `3`. `value <<= tensor_view` is the load spelling;
`value >>= tensor_view` stores, while a scalar right operand retains arithmetic
right shift. [Scalar memory](../../examples/api/scalar_memory) checks
both spellings, an independent copied cell and untouched UB lanes through all
four facades. Each memory consumer still needs a supported dtype/backend.
A launch that supplies initialized
output for a read/modify/write kernel uses `seed_outputs=True`; normal output
checks use poison values so a missing write fails.

`CeilDiv`, `Align8` through `Align256`, `Min`/`Max`, `scalar_abs`/`scalar_sqrt`,
`var_*` and worker queries cover scalar tile calculations. Their result types
depend on operands and scope. Do not infer negative division/remainder or
mixed-width behavior from old Python tracing descriptions, or confuse scalar
`Min`/`Max` with A2 tensor or A5 register operations. Each example declares its
tested input domain. [Scalar arithmetic](../../examples/api/scalar_math)
observes integer operators, float division/square root, alignment and worker
queries; literal float division retains its fractional result.

[Dynamic scalar absolute value](../../examples/api/scalar_abs)
checks i8/i16/i32/i64/f32 and FP16 on A5. CCE and PTO ISA use the native
scalar instruction, including conversion of negative zero to positive zero.
The current PyPTO Pro language's ordinary scalar abs/sqrt/cast restrictions
are explicit upstream gaps, distinct from supported Tile/VF/SIMT operations
([upstream scalar restrictions](../upstream.md#a5-up-001)).

The measured A5 scalar square-root paths include FP32, FP16 and BF16, with a
separate FP32 SIMT path. Ordinary BF16 abs/sqrt and typed scalar conversions
use explicit widening/narrowing IR and the SDK's `AscendC::Cast` overloads on
CCE/PTO. Tests cover signed zero, subnormals, ties, infinities and NaN classes,
including every BF16 input bit pattern (M10-055).
BF16 register conversion and matrix arithmetic have separate contracts.

Typed integer division and remainder use floor semantics. Use
`var_div(a, b, rounding="trunc")` and `var_mod(a, b, rounding="trunc")` for
explicit truncation toward zero. `CeilDiv` handles either divisor sign without
an overflowing `a + b - 1` intermediate. Division by zero and signed minimum
divided by -1 are invalid; static checks and the models diagnose them.

A5 synchronization has three separate limits: intra-core raw event IDs 0..7,
public logical cross-core IDs 0..10, and at most 15 pending counts per cross-core
ID. The mode-4 AIV1 encoding adds 16 internally; callers continue to pass 0..10.

Legacy `VarList`/`VarRef` are absent from the agreed facade. A static Python
container of cells can organize statically selected values; it is not an
advertised replacement for the old dynamically indexed scalar array.

The public authoring regressions check entry invocation, device binding,
unsupported legacy names, loop diagnostics and thread-count boundaries in
`tests/product/test_api_authoring.py`. Numerical and pipe checks are separate
example commands; their current artifact pins and outcomes appear in the
migration evidence — historical record (`docs/migration/fragments/library-examples.progress.json`).

The integer division regressions
cover the repaired floor/truncation contract (M10-047). The
native boundary probes retain
their separately qualified scalar dtype/backend scope.
